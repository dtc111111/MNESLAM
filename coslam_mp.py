import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["OMP_NUM_THREADS"] = '3'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import time

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from datasets.dataset_track import get_dataset_track
from utils import coordinates, extract_mesh
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion, cam_pose_to_matrix

# Multiprocessing imports
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
        # self.is_optimizing = SLAM.is_optimizing

        # backend process
        self.backend = Backend(self.net, self.video, self.config, self.device)

    def info(self, msg):
        print(Fore.GREEN)
        print(msg)
        print(Style.RESET_ALL)

    def forward(self):
        cur_t = self.video.counter.value
        t = cur_t
        # print('bbbbbbbbbbbbaaaaaaaaaaaaaaaa',cur_t)
        if cur_t > self.frontend_window:

            t_start = 0
            now = f'{strftime("%Y-%m-%d %H:%M:%S", gmtime())} - Full BA'
            msg = f'\n\n {now} : [{t_start}, {t}]; Current Keyframe is {cur_t}, last is {self.last_t}.'

            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False)
            self.info(msg+'\n')

            self.last_t = cur_t



class CoSLAM():
    def __init__(self, config):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        self.create_bounds()
        # self.create_pose_data()
        self.get_pose_representation()

        self.create_share_data()

        self.dataset_track = get_dataset_track(config, device=self.device)

        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device).share_memory()
        self.create_optimizer()

        self.net = DroidNet()

        self.load_pretrained(config['tracking']['pretrained'])
        self.net.to(self.device).eval()
        self.net.share_memory()
        # self.mesher = Mesher(config, self)
        # store images, depth, poses, intrinsics (shared between process)
        self.video = DepthVideo(config)
        self.tracker = Tracker(config, self)
        self.mapper = Mapper(config, self)


        self.traj_filler = PoseTrajectoryFiller(net=self.net, video=self.video, device=self.device)

############################################################

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
        self.is_optimizing = torch.zeros((1)).share_memory_()
        ###########################
        self.all_triggered = torch.zeros((1)).int().share_memory_()
        self.hang_on = torch.zeros((1)).int().share_memory_()
        self.hang_on_mesh = torch.zeros((1)).int().share_memory_()
        self.tracking_finished = torch.zeros((1)).int().share_memory_()
        self.mapping_finished = torch.zeros((1)).int().share_memory_()
        self.num_running_thread = torch.zeros((1)).int().share_memory_()
        self.optimizing_finished = torch.zeros((1)).int().share_memory_()
    
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
        print('#kf:', num_kf)
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
        save_dict = {'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

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
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        # if sdf:
        #     loss += ret["sdf_losses"]
        if smooth:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'],

                                                                                margin=self.config['training']['smooth_margin'])
        
        return loss             


    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

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
                                # {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
                                {'params': planes_para, 'eps': 1e-15,
                                 'lr': self.config['mapping']['lr_embed']}]  # lr modify?

        if not self.config['grid']['oneGrid']:
            # trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
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
                                     'mapping_vis')
        os.makedirs(f'{img_savepath}', exist_ok=True)

        with torch.no_grad():

            gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
            gt_color_np = gt_color.squeeze(0).cpu().numpy()

            if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
            else:
                c2w = c2w_or_camera_tensor.squeeze().detach()

            # self, c2w, device, gt_depth=None
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

    def mapping(self, rank):
        print('Mapping Triggered!')
        self.all_triggered += 1

        while (self.all_triggered < self.num_running_thread):
            pass

        while (self.tracking_finished < 1 or self.video.map_counter.value < self.video.counter.value)and(not self.is_optimizing):
            # while (self.hang_on > 0 or self.hang_on_mesh > 0):
            #     time.sleep(1.0)
            #     if self.is_optimizing = 1:
            #         self.mapper.run()
            self.mapper.run()

        ###########################3最后map?
        self.mapping_finished += 1
        print('Mapping Done!')

    def tracking(self, rank):
        self.all_triggered += 1  # Confirm the initiation of all threads
        while (self.all_triggered < self.num_running_thread):
            pass

        while True:
            ########finishmapfirst
            if self.mapping_first_frame[0] == 1:
                break
        #     time.sleep(0.5)

        # self.tracker.run()
        print('Start tracking')

        for (timestamp, image, depth, intrinsic, gt_pose) in tqdm(self.dataset_track):
            ############## wait or not
            # while self.video.map_counter.value < self.video.counter.value:
            #     time.sleep(0.1)
            # print(self.video.timestamp)
            self.tracking_idx[0] = timestamp
            self.tracker.run(timestamp, image, depth, intrinsic, gt_pose)

            # if timestamp % self.config['vis']['render_freq'] == 0 and timestamp > 0:
            #     self.hang_on[:] = 1
            #
            # if timestamp % self.config['vis']['mesh_freq'] == 0 and timestamp > 0:
            #     self.hang_on_mesh[:] = 1

            # while(self.hang_on > 0 or self.hang_on_mesh > 0):
            #     time.sleep(1.0)

            # torch.cuda.empty_cache()

        self.tracking_finished += 1

        print('Tracking Done!')

    def optimizing(self, rank):
        print('Full Bundle Adjustment Triggered!')
        self.all_triggered += 1
        while(self.tracking_finished < 1):
            # while(self.hang_on > 0 and self.hang_on_mesh > 0):
            #     time.sleep(0.5)

            self.ba()

        self.ba()
        self.optimizing_finished += 1

        print('Full Bundle Adjustment  Done!')

    # def visualizing(self, rank):
    #     print('Visualization Triggered!')
    #     self.all_triggered += 1
    #
    #     while(self.tracking_finished < 1 or self.optimizing_finished < 1):
    #         while self.hang_on < 1:
    #             time.sleep(1.0)
    #
    #         idx = int(self.tracking_idx[0].item())
    #         print('vvvvvvvvvvvvvvvvisssssssssssssss', idx)
    #         batch = self.dataset[idx]
    #
    #         #################################
    #         self.save_imgs(idx, batch['depth'], batch['rgb'], batch['c2w'])
    #
    #         self.hang_on[:] = 0
    #
    #     self.visualizing_finished += 1
    #     print('Visualization Done!')


    # def meshing(self, rank):
    #     print('Meshing Triggered!')
    #     self.all_triggered += 1
    #
    #     while self.mapping_finished < 1 :
    #         while(self.hang_on_mesh < 1):
    #             time.sleep(1.0)
    #
    #         self.save_mesh(self.tracking_idx[0], voxel_size=self.config['mesh']['voxel_eval'])
    #         self.hang_on_mesh[:] = 0
    #
    #     self.meshing_finished += 1
    #     print('Meshing Done!')

    def terminate(self, rank):
        """ fill poses for non-keyframe images and evaluate """

        # while(self.optimizing_finished < 1):
        #
        #     print(self.num_running_thread, self.tracking_finished)
        #     if self.num_running_thread == 1 and self.tracking_finished > 0:
        #         break

        # self.mapper.final_run()
        timestamp_log = self.video.timestamp.to('cpu').tolist()

        # 打开一个文件并写入 Tensor 数据，元素之间以空格分隔
        with open(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'timestamp_log.txt'), 'w') as f:
            f.write(' '.join(f"{item}" for item in timestamp_log)) # 将所有元素转换为字符串并以空格连接


        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                      'final_checkpoint{}.pt'.format(int(self.tracking_idx[0])))

        # self.video

        self.save_ckpt(model_savepath)
        self.save_mesh(int(self.tracking_idx[0]), voxel_size=self.config['mesh']['voxel_final'])
        #
        # mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
        # mesh_directory = os.path.join(mesh_save_path, 'mesh')
        #
        # # Create the directory if it doesn't exist
        # if not os.path.exists(mesh_directory):
        #     os.makedirs(mesh_directory)
        #
        # mesh_out_file = os.path.join(mesh_directory, 'final_mesh.ply')
        # self.mesher.get_mesh(mesh_out_file, self.mapper.keyframe_dict, self.device)


        # pose_evaluation(self.pose_gt, self.est_c2w_data, 1,
        #                 os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)

        timestamps = [i for i in range(len(self.dataset_track))]
        camera_trajectory = self.traj_filler(self.dataset_track)  # w2cs
        w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
        camera_trajectory = w2w * camera_trajectory.inv()
        traj_est = camera_trajectory.data.cpu().numpy()
        estimate_c2w_list = camera_trajectory.matrix().data.cpu()

        pose_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
        np.save(
            f'{pose_save_path}/est_poses.npy',
            estimate_c2w_list.numpy(),  # c2ws
        )

        do_evaluation = True
        if do_evaluation:
            from evo.core.trajectory import PoseTrajectory3D
            import evo.main_ape as main_ape
            from evo.core.metrics import PoseRelation
            from evo.core.trajectory import PosePath3D

            traj_ref = []
            traj_est_select = []
            if self.dataset_track.poses is None:  # for eth3d submission
                if self.dataset_track.image_timestamps is not None:
                    submission_txt = f'{pose_save_path}/submission.txt'
                    with open(submission_txt, 'w') as fp:
                        for tm, pos in zip(self.dataset_track.image_timestamps, traj_est.tolist()):
                            str = f'{tm:.9f}'
                            for ps in pos:  # timestamp tx ty tz qx qy qz qw
                                str += f' {ps:.14f}'
                            fp.write(str+'\n')
                    print('Poses are save to {}!'.format(submission_txt))

                print("Terminate: no GT poses found!")
                trans_init = None
                gt_c2w_list = None
            else:
                for i in range(len(self.dataset_track.poses)):
                    val = self.dataset_track.poses[i].sum()
                    if np.isnan(val) or np.isinf(val):
                        print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
                        continue
                    traj_est_select.append(traj_est[i])
                    traj_ref.append(self.dataset_track.poses[i])

                traj_est = np.stack(traj_est_select, axis=0)
                gt_c2w_list = torch.from_numpy(np.stack(traj_ref, axis=0))

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:,:3],
                    orientations_quat_wxyz=traj_est[:,3:],
                    timestamps=np.array(timestamps))

                traj_ref =PosePath3D(poses_se3=traj_ref)

                result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                                      pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

                out_path=f'{pose_save_path}/metrics_traj.txt'
                with open(out_path, 'a') as fp:
                    fp.write(result.pretty_str())
                trans_init = result.np_arrays['alignment_transformation_sim3']

                print('trans_init', trans_init)

            # if self.meshing_finished > 0 and (not self.only_tracking):
            #     self.mesher(the_end=True, estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list, trans_init=trans_init)

        print("Terminate: Done!")

    def run(self):
        processes = [
            mp.Process(target=self.mapping, args=(0,)),
            mp.Process(target=self.tracking, args=(1,)),
            mp.Process(target=self.optimizing, args=(2,)),  # ba
            # mp.Process(target=self.visualizing, args=(2,)),
            # mp.Process(target=self.meshing, args=(3, )),  # only generate mesh at the very end

        ]

        self.num_running_thread[0] += len(processes)

        # for rank in range(4):
        #     if rank == 1:
        #         p = mp.Process(target=self.tracking, args=(rank, ))
        #     # elif rank == 2:
        #     #     p = mp.Process(target=self.optimizing, args=(rank,))
        #     elif rank == 2:
        #         p = mp.Process(target=self.visualizing, args=(rank,))
        #     elif rank == 3:
        #         p = mp.Process(target=self.meshing, args=(rank,))
        #
        #     elif rank == 0:
        #         p = mp.Process(target=self.mapping, args=(rank, ))
        #         time.sleep(2)
        #
        #     p.start()
        #     processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()



if __name__ == '__main__':
            
    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))

    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = CoSLAM(cfg)

    slam.run()

    slam.terminate(rank=-1)
