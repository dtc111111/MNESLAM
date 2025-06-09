import os

import torch
import torch.optim as optim
import random
import argparse
import json

from matplotlib import pyplot as plt
from tqdm import tqdm

import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from datasets.dataset_track import get_dataset_track
from utils import coordinates, extract_mesh
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion, cam_pose_to_matrix
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from mp_slam.tracker import Tracker
from mp_slam.mapper import Mapper

import torch.nn as nn

from tracker.droid_net import DroidNet
from collections import OrderedDict
from tracker.depth_video import DepthVideo
from tracker.trajectory_filler import PoseTrajectoryFiller
from lietorch import SE3
import numpy as np

from tracker.backend import Backend
from colorama import Fore, Style
from time import gmtime, strftime
from model.Mesher import Mesher
import multiprocessing

from PIL import Image


class BundleAdjustment(nn.Module):
    def __init__(self, config, SLAM):
        super(BundleAdjustment, self).__init__()
        self.args = args
        self.config = config
        self.device = SLAM.device
        self.net = SLAM.net
        self.video = SLAM.video

        self.frontend_window = config['tracking']['frontend']['window']
        self.last_t = -1
        self.ba_counter = -1

        # backend process
        self.backend = Backend(self.net, self.video, self.config, self.device)

    def info(self, msg):
        print(Fore.GREEN)
        print(msg)
        print(Style.RESET_ALL)

    def forward(self):
        cur_t = self.video.counter.value
        t = cur_t

        if cur_t > self.frontend_window:

            t_start = 0
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Full BA'
            msg = f'\n\n {now} : [{t_start}, {t}]; Current Keyframe is {cur_t}, last is {self.last_t}.'

            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False)
            self.info(msg+'\n')

            self.last_t = cur_t


class MNESLAM():
    def __init__(self, config):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        self.create_bounds()
        self.get_pose_representation()

        self.create_share_data()

        self.dataset_track = get_dataset_track(config, device=self.device)

        self.keyframeDatabase = self.create_kf_database(config)

        self.model = JointEncoding(config, self.bounding_box).to(self.device).share_memory()

        self.model_shared = JointEncoding(config, self.bounding_box).to(self.device).share_memory()
        for param in self.model_shared.parameters():
            param.requires_grad = False

        self.create_optimizer()

        self.net = DroidNet()

        self.load_pretrained(config['tracking']['pretrained'])
        self.net.to(self.device).eval()
        self.net.share_memory()

        self.mesher = Mesher(config, self)

        # store images, depth, poses, intrinsics (shared between process)
        self.video = DepthVideo(config, self)

        self.tracker = Tracker(config, self)
        self.mapper = Mapper(config, self)

        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)

        self.ba = BundleAdjustment(config,  self)

    def load_pretrained(self, pretrained):
        print(f'INFO: load pretrained checkpiont from {pretrained}!')

        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(pretrained).items()
        ])

        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]

        self.net.load_state_dict(state_dict)

    def pose_eval_func(self):
        return pose_evaluation
    
    def create_share_data(self):
        # self.create_pose_data()
        self.mapping_first_frame = torch.zeros((1)).int().share_memory_()
        self.mapping_idx = torch.zeros((1)).share_memory_()
        self.tracking_idx = torch.zeros((1)).share_memory_()
        self.vis_timestamp = torch.zeros((1)).share_memory_()

        ###########################
        self.all_triggered = torch.zeros((1)).int().share_memory_()
        self.hang_on = torch.zeros((1)).int().share_memory_()
        self.hang_on_mesh = torch.zeros((1)).int().share_memory_()
        self.tracking_finished = torch.zeros((1)).int().share_memory_()
        self.mapping_finished = torch.zeros((1)).int().share_memory_()
        self.num_running_thread = torch.zeros((1)).int().share_memory_()
        self.optimizing_finished = torch.zeros((1)).int().share_memory_()

        manager = multiprocessing.Manager()
        self.keyframe_dict = manager.list()
    
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        num_frames = self.dataset.num_frames
        self.est_c2w_data = torch.zeros((num_frames, 4, 4)).to(self.device).share_memory_()
        self.est_c2w_data_rel = torch.zeros((num_frames, 4, 4)).to(self.device).share_memory_()
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''

        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = torch.zeros((self.dataset.num_frames, 4, 4))
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        keyframe_list = list(self.keyframe_dict)

        save_dict = {
            'all_planes': self.model.all_planes,
            'model': self.model.state_dict(),
            'keyframes': keyframe_list,
            'bound': self.bounding_box
        }

        torch.save(save_dict, save_path)
        print('Checkpoint saved successfully.')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, is_co_sdf=True, depth=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            if is_co_sdf:
                co_loss = self.config['training']['sdf_weight'] * ret["co_sdf_loss"] + self.config['training']['fs_weight'] * ret["co_fs_loss"]
                loss += co_loss
            else:
                e_loss = self.config["mapping"]["w_sdf_fs"] * ret["e_fs_loss"] + self.config["mapping"]["w_sdf_center"] * \
                        ret["e_center_loss"]+ self.config["mapping"]["w_sdf_tail"] * ret["e_tail_loss"]
                loss += e_loss

        return loss             

    def get_rays_from_batch(self, batch, c2w_est, indices):
        '''
        Get the rays from the batch
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
            c2w_est: [4, 4]
            indices: [N]
        Returns:
            rays_o: [N, 3]
            rays_d: [N, 3]
            target_s: [N, 3]
            target_d: [N, 1]
            c2w_gt: [4, 4]
        '''
        rays_d_cam = batch['direction'].reshape(-1, 3)[indices].to(self.device)
        target_s = batch['rgb'].reshape(-1, 3)[indices].to(self.device)
        target_d = batch['depth'].reshape(-1, 1)[indices].to(self.device)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:3, :3], -1)
        rays_o = c2w_est[None, :3, -1].repeat(rays_d.shape[0], 1)
        c2w_gt = batch['c2w'][0].to(self.device)

        if torch.sum(torch.isnan(rays_d_cam)):
            print('warning rays_d_cam')
        
        if torch.sum(torch.isnan(c2w_est)):
            print('warning c2w_est')

        return rays_o, rays_d, target_s, target_d, c2w_gt

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # Optimizer for BA

        planes_para = []

        if not self.config['grid']['oneGrid']:
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = self.model.all_planes
            c_planes_para = []

            print(self.planes_xy is self.model.all_planes[0])  # True 表示它们是同一个对象

            for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
                for i, c_plane in enumerate(c_planes):
                    c_plane = nn.Parameter(c_plane)
                    c_planes_para.append(c_plane)
                    c_planes[i] = c_plane
        else:
            self.planes_xy, self.planes_xz, self.planes_yz = self.model.all_planes

        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane

        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6,
                                 'lr': self.config['mapping']['lr_decoder']},

                                {'params': planes_para, 'eps': 1e-15,
                                 'lr': self.config['mapping']['lr_embed']}]  # lr modify?

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append(
                {'params': c_planes_para, 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))


    def save_imgs(self, idx, gt_depth, gt_color, c2w_or_camera_tensor):
        """
        Visualization of depth and color images and save to file.
        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in
                camera to world matrix or quaternion and translation tensor.
            all_planes (Tuple): feature planes.
            all_planes_global (Tuple): global feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
        """
        img_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                     'eval_vis')
        os.makedirs(f'{img_savepath}', exist_ok=True)


        with torch.no_grad():

            if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
            else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

            depth, color = self.model.render_img(c2w, self.device, gt_depth=None)
            color_np = color.detach().cpu().numpy()
            color_np = np.clip(color_np, 0, 1)
            color_np_uint8 = (color_np * 255).astype(np.uint8)
            image = Image.fromarray(color_np_uint8)
            image.save(f'{img_savepath}/{idx:05d}.jpg')

            gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
            gt_color_np = gt_color.squeeze(0).cpu().numpy()

            if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
            else:
                c2w = c2w_or_camera_tensor.squeeze().detach()

            depth, color = self.model.render_img(c2w, self.device, gt_depth=gt_depth)
            depth_np = depth.detach().cpu().numpy()
            color_np = color.detach().cpu().numpy()
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[gt_depth_np == 0.0] = 0.0
            color_residual = np.abs(gt_color_np - color_np)
            color_residual[gt_depth_np == 0.0] = 0.0

            fig, axs = plt.subplots(2, 3)
            fig.tight_layout()
            max_depth = np.max(gt_depth_np)

            axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 0].set_title('Input Depth')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 1].set_title('Generated Depth')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
            axs[0, 2].set_title('Depth Residual')
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])
            gt_color_np = np.clip(gt_color_np, 0, 1)
            color_np = np.clip(color_np, 0, 1)
            color_residual = np.clip(color_residual, 0, 1)
            axs[1, 0].imshow(gt_color_np, cmap="plasma")
            axs[1, 0].set_title('Input RGB')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].imshow(color_np, cmap="plasma")
            axs[1, 1].set_title('Generated RGB')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual')
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
            plt.subplots_adjust(wspace=0, hspace=0)

            plt.savefig(f'{img_savepath}/{idx:05d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
            plt.cla()
            plt.clf()

    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)       
        
    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                           {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer

    def run(self):
        processes = [
            mp.Process(target=self.mapping, args=(0,)),
            mp.Process(target=self.tracking, args=(1,)),
            mp.Process(target=self.optimizing, args=(2,)),
        ]

        self.num_running_thread[0] += len(processes)

        for p in processes:
            p.start()

        for p in processes:
            p.join()




