import os
os.environ["OMP_NUM_THREADS"] = '3'

# Package imports
import torch
import torch.optim as optim
import random
import argparse
import shutil
import json

from matplotlib import pyplot as plt
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
import multiprocessing

from PIL import Image
from threading import Thread


class BundleAdjustment(nn.Module):
    def __init__(self, config, SLAM):
        super(BundleAdjustment, self).__init__()
        # self.args = args
        self.config = config
        self.device = SLAM.device
        self.net = SLAM.net
        self.video = SLAM.video

        self.frontend_window = config['tracking']['frontend']['window']
        self.last_t = -1
        self.ba_counter = -1

        # backend process
        self.backend = Backend(self.net, self.video, self.config, self.device)

        self.rank = SLAM.rank

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
            msg = f'\n\n {now} : [{t_start}, {t}]; Current Keyframe is {cur_t}, last is {self.last_t}, agent{self.rank}.'

            self.backend.dense_ba(t_start=t_start, t_end=t, steps=6, motion_only=False)
            self.info(msg+'\n')

            self.last_t = cur_t


class MNESLAM():
    def __init__(self, config, rank=0, world_size=1, shared_components=None):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = get_dataset(config)
        self.create_bounds()
        self.all_agent_bounds = self.load_all_agent_bounds(config, rank, world_size)
        # self.create_pose_data()
        self.get_pose_representation()

        # Use shared components if provided, otherwise create them
        self.create_share_data(shared_components)

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
        
        print(f"Agent {self.rank} initialized on device {self.device}.")

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
    
    def create_share_data(self, shared_components=None):
        if shared_components:
            # For multi-agent, only the descriptor DB is shared
            self.descriptor_db_lock = shared_components['descriptor_db_lock']
            self.descriptor_db = shared_components['descriptor_db']
        else:
            # For single agent, create its own descriptor DB for consistency
            self.descriptor_db_lock = multiprocessing.Lock()
            manager = multiprocessing.Manager()
            self.descriptor_db = manager.list()

        # Keyframe dictionary is now ALWAYS local to the agent
        self.keyframe_dict_lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        self.keyframe_dict = manager.list()

        # These should be local states for each agent, not shared across processes.
        self.mapping_first_frame = torch.zeros((1), dtype=torch.int)
        self.mapping_idx = torch.zeros((1))
        self.tracking_idx = torch.zeros((1))
        self.vis_timestamp = torch.zeros((1))

        # Control flags should be local to each SLAM instance (process)
        self.all_triggered = torch.zeros((1), dtype=torch.int)
        self.tracking_finished = torch.zeros((1), dtype=torch.int)
        self.mapping_finished = torch.zeros((1), dtype=torch.int)
        self.num_running_thread = torch.zeros((1), dtype=torch.int)
        self.optimizing_finished = torch.zeros((1), dtype=torch.int)
    
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
        #######change
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        # print('#kf:', num_kf)
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
 
    def load_all_agent_bounds(self, main_config, current_rank, world_size):
        """
        Loads bounding boxes for all agents from the main config's 'loop_bound' section.
        Args:
            main_config (dict): The main configuration dictionary.
            current_rank (int): The rank of the current agent (0-indexed).
            world_size (int): The total number of agents.
        Returns:
            dict: A dictionary where keys are agent ranks (0-indexed) and values are their
                  bounding box tensors.
        """
        if world_size == 1:
            # Single agent, use its own standard bound.
            # The key is the agent's rank.
            return {current_rank: torch.from_numpy(np.array(main_config['mapping']['bound']))}

        all_bounds = {}
        loop_bounds_config = main_config.get('loop_bound')
        default_bound_np = np.array(main_config['mapping']['bound'])

        if loop_bounds_config is None:
            print(f"[WARN] Agent {current_rank}: 'loop_bound' section not found in the main config. "
                  f"Using default bound for all {world_size} agents.")
            for rank in range(world_size):
                all_bounds[rank] = torch.from_numpy(default_bound_np)
            return all_bounds

        for rank in range(world_size):
            bound_key = f'bound_{rank}'
            if bound_key in loop_bounds_config:
                bound_data = loop_bounds_config[bound_key]
                all_bounds[rank] = torch.from_numpy(np.array(bound_data))
            else:
                print(f"[WARN] Agent {current_rank}: Bound for agent {rank} ('{bound_key}') not found in 'loop_bound'. "
                      f"Using default bound.")
                all_bounds[rank] = torch.from_numpy(default_bound_np)

        return all_bounds

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_latest_checkpoint(self):
        """
        Saves the current model state to a 'latest_checkpoint.pt' for other agents to use.
        This enables online map fusion.
        """
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
        os.makedirs(agent_output_dir, exist_ok=True)
        save_path = os.path.join(agent_output_dir, 'latest_checkpoint.pt')

        # For distillation, we primarily need the model's parameters.
        save_dict = {
            'model': self.model.state_dict(),
            'all_planes': self.model.all_planes,
            'bound': self.model.bound.cpu(),
            'bounding_box': self.model.bounding_box.cpu(),
        }

        # Atomic save: write to a temporary file then rename. This prevents another
        # process from reading a partially written checkpoint.
        temp_save_path = save_path + ".tmp"
        torch.save(save_dict, temp_save_path)
        os.replace(temp_save_path, save_path)

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        with self.keyframe_dict_lock:
            # 将 keyframe_dict 转换为普通的 Python 列表
            keyframe_list = list(self.keyframe_dict)

        save_dict = {
            'all_planes': self.model.all_planes,
            'model': self.model.state_dict(),
            # 'keyframes': keyframe_list, # This is now a safe copy
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
        img_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],  f'agent_{self.rank}',
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

        # with torch.no_grad():

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
            plt.close()

    def save_mesh(self, agent_dir, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], agent_dir, 'mesh', 'mesh_track_{}.ply'.format(i))
        # mesh_savepath = os.path.join(agent_output_dir, 'mesh_track{}.ply'.format(i))
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

        while self.tracking_finished < 1 or self.video.map_counter.value < self.video.counter.value-1:
            self.mapper.run()

        while self.video.map_counter.value < self.video.counter.value:
            self.mapper.final_run()
        self.mapping_finished += 1
        print('Mapping Done!')

    def tracking(self, rank):
        self.all_triggered += 1  # Confirm the initiation of all threads
        while (self.all_triggered < self.num_running_thread):
            pass

        while True:
            if self.mapping_first_frame[0] == 1:
                break
        print('Start tracking')

        for (timestamp, image, depth, intrinsic, gt_pose) in tqdm(self.dataset_track):
            self.tracking_idx[0] = timestamp
            self.tracker.run(timestamp, image, depth, intrinsic, gt_pose)

            torch.cuda.empty_cache()

        self.tracking_finished += 1

        print('Tracking Done!')

    def optimizing(self, rank):
        print('Full Bundle Adjustment Triggered!')
        self.all_triggered += 1
        # while(self.tracking_finished < 1)or(self.mapping_finished<1):
        while self.tracking_finished < 1:
            self.ba()

        self.ba()
        self.optimizing_finished += 1

        print('Full Bundle Adjustment  Done!')

    def terminate(self, rank):
        """ fill poses for non-keyframe images and evaluate """

        # Add agent rank to output files to avoid conflicts
        output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
        agent_output_dir = os.path.join(output_dir, f'agent_{self.rank}')
        os.makedirs(agent_output_dir, exist_ok=True)

        model_savepath = os.path.join(agent_output_dir, 'final_checkpoint.pt')

        self.save_ckpt(model_savepath)
        self.save_mesh(f'agent_{self.rank}', 'final', voxel_size=self.config['mesh']['voxel_final'])

        mesh_out_file = os.path.join(agent_output_dir, 'final_mesh.ply')
        print('Saving final mesh!')
        with self.keyframe_dict_lock:
            keyframe_dict_on_cpu = list(self.keyframe_dict)
            keyframe_dict_on_gpu = []
            for keyframe in keyframe_dict_on_cpu:
                keyframe_dict_on_gpu.append({
                    'color': keyframe['color'].to(self.device),
                    'depth': keyframe['depth'].to(self.device),
                    'est_c2w': keyframe['est_c2w'].to(self.device)
                })
        self.mesher.get_mesh(mesh_out_file, keyframe_dict_on_gpu, self.device)

        timestamps = [i for i in range(len(self.dataset_track))]
        camera_trajectory = self.traj_filler(self.dataset_track)  # w2cs

        w2w = SE3(self.video.pose_compensate[0].clone().unsqueeze(dim=0)).to(camera_trajectory.device)
        camera_trajectory = w2w * camera_trajectory.inv()
        traj_est = camera_trajectory.data.cpu().numpy()
        estimate_c2w_list = camera_trajectory.matrix().data.cpu().numpy()

        np.save(
            f'{agent_output_dir}/est_poses.npy',
            estimate_c2w_list,  # c2ws
        )
        
        num_keyframes = self.video.counter.value
        keyframe_poses = self.video.get_all_pose(self.device)[:num_keyframes].cpu().numpy()
        np.save(
            f'{agent_output_dir}/key_est_poses.npy',
            keyframe_poses,  # c2ws
        )

        # Also save the corresponding timestamps, which are crucial for alignment
        keyframe_timestamps = self.video.timestamp[:num_keyframes].cpu().numpy()
        np.save(
            f'{agent_output_dir}/key_timestamps.npy',
            keyframe_timestamps
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
                if self.dataset_track.image_timestamps is not None: # This part might need adjustment for multi-agent
                    submission_txt = f'{agent_output_dir}/submission.txt'
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
                print(f"Agent {self.rank} Evaluation Results:\n" + result.pretty_str())
                out_path=f'{agent_output_dir}/metrics_traj.txt'
                with open(out_path, 'a') as fp:
                    fp.write(result.pretty_str())
                trans_init = result.np_arrays['alignment_transformation_sim3']

                print('trans_init', trans_init)

        print("Terminate: Done!")

    def run(self):
        t1 = Thread(target=self.mapping, args=(0,))
        t2 = Thread(target=self.tracking, args=(1,))
        t3 = Thread(target=self.optimizing, args=(2,))

        self.num_running_thread[0] += 3

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

def run_agent(rank, world_size, config, shared_components):
    from multiprocessing.resource_tracker import unregister
    """
    The main function for each agent process.
    """
    
    # Each agent has its own SLAM instance
    slam = MNESLAM(config, rank, world_size, shared_components)
    
    # Start the internal threads for tracking, mapping, optimizing
    slam.run()
    
    # Wait for SLAM to finish and then terminate
    slam.terminate(rank=rank)

    print(f"Agent {rank} has terminated.")