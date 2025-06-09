# Use dataset object
import numpy as np
import torch
import time
import os
import random
from optimization.utils import homogeneous_matrix_to_pose
import torch.nn.functional as F

from datasets.utils import get_camera_rays

class Mapper():
    def __init__(self, config, SLAM) -> None:
        self.config = config
        self.slam = SLAM
        self.model = SLAM.model
        self.model_shared = SLAM.model_shared
        self.tracking_idx = SLAM.tracking_idx
        self.mapping_idx = SLAM.mapping_idx
        self.mapping_first_frame = SLAM.mapping_first_frame

        self.map_optimizer = SLAM.map_optimizer
        self.device = SLAM.device
        self.dataset = SLAM.dataset
        self.video = SLAM.video
        self.keyframe_dict = SLAM.keyframe_dict
        self.mesher = SLAM.mesher

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = SLAM.H, SLAM.W, SLAM.fx, SLAM.fy, SLAM.cx, SLAM.cy

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print('First frame mapping...')
        if batch['frame_id'] != 0:
            raise ValueError('First frame mapping must be the first frame!')
        c2w = batch['c2w'].to(self.device)

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.slam.select_samples(self.slam.dataset.H, self.slam.dataset.W, self.config['mapping']['sample'])
            indice_h, indice_w = indice % (self.slam.dataset.H), indice // (self.slam.dataset.H)
            rays_d_cam = batch['direction'][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o.to(self.device), rays_d.to(self.device), target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret, is_co_sdf=self.config['is_co_sdf'])
            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.video.keyframe.add_keyframe(batch, 1, filter_depth=self.config['mapping']['filter_depth'])
        # if self.config['mapping']['first_mesh']:
        self.keyframe_dict.append(
            {'color': batch['rgb'].cpu(), 'depth': batch['depth'].cpu(),
             'est_c2w': c2w.clone().cpu()})
        
        print('First frame mapping done')
        self.mapping_first_frame[0] = 1
        self.slam.save_imgs(0, batch['depth'], batch['rgb'], batch['c2w'])
        self.video.map_counter.value += 1
        return ret, loss

    def global_BA(self, batch, poses):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''

        # Set up optimizer
        self.map_optimizer.zero_grad()
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.video.keyframe.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W), max(self.config['mapping']['sample'] // len(self.video.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat([ids, -torch.ones((len(idx_cur)))]).to(torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses[ids_all, None, :3, :3], -1)
            rays_o = poses[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            # loss = self.slam.get_loss_from_ret(ret, smooth=True)
            loss = self.slam.get_loss_from_ret(ret, is_co_sdf=self.config['is_co_sdf'])
            print('loss', loss)
            loss.backward()

            self.map_optimizer.step()
            self.map_optimizer.zero_grad()

    def distillation(self, poses_all):

        for i in range(self.config['mapping']['loop_iters']):
            self.map_optimizer.zero_grad()
            bs = self.config['training']['n_samples']
            indices = torch.randint(0, self.dataset.total_pixels, (bs,))
            rays_d_cam = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy).reshape(-1, 3)[
                indices].to(self.device)
            # N pose Bs rays
            poses_all = torch.stack(
                [torch.tensor(pose, dtype=torch.float32) for pose in poses_all]).to(
                self.device)

            rays_d = torch.sum(rays_d_cam[None, :, None, :] * poses_all[:, None, :3, :3], -1)

            rays_o = poses_all[..., None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)


            agent1_rendered_ret = self.model_shared.render_rays(rays_o, rays_d, target_d=None, loop=True)
            agent1_rendered_rgb, agent1_rendered_depth = agent1_rendered_ret['rgb'], agent1_rendered_ret['depth']

            agent1_rendered_depth = agent1_rendered_depth.unsqueeze(1)

            ret = self.model.forward(rays_o, rays_d, agent1_rendered_rgb, agent1_rendered_depth)
            loss = self.slam.get_loss_from_ret(ret, is_co_sdf=self.config['is_co_sdf'])

            loss.backward()
            self.map_optimizer.step()

        idx = int(self.mapping_idx[0].item())
        print('Save loop mesh!', idx)

    def get_loop_pose(self, cur_c2w, batch)
        best_sdf_loss = None
        thresh = 0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        # 用loop b给loop a的pose初始化
        loop_a_rot, loop_a_trans, pose_optimizer = self.slam.get_pose_param_optim(cur_c2w[None, ...], mapping=False)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            loop_a_c2w_est = self.slam.matrix_from_tensor(loop_a_rot, loop_a_trans)

            indice = self.slam.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2,
                                              self.config['tracking']['sample'])

            # Slicing
            indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
            rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(
                self.device)

            rays_o = loop_a_c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * loop_a_c2w_est[:, :3, :3], -1)

            agent1_rendered_ret = self.model_shared.render_rays(rays_o, rays_d, target_d=None, loop=True)
            agent1_rendered_rgb, agent1_rendered_depth = agent1_rendered_ret['rgb'], agent1_rendered_ret['depth']
            agent1_rendered_depth = agent1_rendered_depth.unsqueeze(1)

            ret = self.model.forward(rays_o, rays_d, agent1_rendered_rgb, agent1_rendered_depth)
            # ret = {'rgb' : rgb_map, 'depth' :depth_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_var':depth_var,}
            loss = self.slam.get_loss_from_ret(ret, sdf=False)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_loop_a_c2w_est = loop_a_c2w_est.detach()

            with torch.no_grad():
                loop_a_c2w_est = self.slam.matrix_from_tensor(loop_a_rot, loop_a_trans)
                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_loop_a_c2w_est = loop_a_c2w_est.detach()
                    thresh = 0
                else:
                    thresh += 1

            if thresh > self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()

        # b 坐标系下 a 的位姿
        loop_a_pose = best_loop_a_c2w_est.detach().clone()[0]
        # 从 b 坐标系到 a 坐标系的变换矩阵
        loop_b2a_pose = loop_a_pose @ cur_c2w.inverse()

    def run(self):

        # Start mapping
        # while self.tracking_idx[0] < len(self.dataset)-1:
        if self.video.map_counter.value == 0:
            batch = self.dataset[0]
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            time.sleep(0.1)

        else:
            while (self.video.counter.value <= self.config['tracking'][
                'warmup'] or self.video.map_counter.value >= self.video.counter.value - 2) and (
                    self.slam.tracking_finished < 1):
                time.sleep(0.1)

            with self.video.get_lock():
                self.video.map_counter.value += 1

                self.N = self.video.map_counter.value
                keyframe_ids = self.video.timestamp[:self.N]

                current_map_id = int(keyframe_ids[-1])
                batch = self.dataset[current_map_id]
                poses = self.video.get_pose(self.N, self.device)
                cur_c2w = poses[-1]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, ...]
                else:
                    batch[k] = torch.tensor([v])
            self.global_BA(batch, poses)
            self.mapping_idx[0] = current_map_id


            self.video.keyframe.add_keyframe(batch, self.video.map_counter.value)

            idx = int(self.mapping_idx[0].item())

            self.slam.save_imgs(idx, batch['depth'][0], batch['rgb'][0], cur_c2w)
            self.keyframe_dict.append(
                {'color': batch['rgb'][0].cpu(), 'depth': batch['depth'][0].cpu(),
                 'est_c2w': cur_c2w.clone().cpu()})

            pose_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
            np.save(
                f'{pose_save_path}/key_est_poses.npy',
                poses.cpu().numpy(),  # c2ws
            )

            if self.video.map_counter.value % 50 == 0 and self.video.map_counter.value > 0:
                if self.config['is_co_sdf']:
                    self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])
                else:
                    print('save_eslam_mesh!')
                    mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
                    mesh_directory = os.path.join(mesh_save_path, 'mesh')

                    # Create the directory if it doesn't exist
                    if not os.path.exists(mesh_directory):
                        os.makedirs(mesh_directory)

                    mesh_out_file = os.path.join(mesh_directory, f'{idx:05d}_mesh.ply')

                    keyframe_dict_on_cpu = self.keyframe_dict

                    keyframe_dict_on_gpu = []
                    for keyframe in keyframe_dict_on_cpu:
                        keyframe_dict_on_gpu.append({
                            'color': keyframe['color'].to(self.device),
                            'depth': keyframe['depth'].to(self.device),
                            'est_c2w': keyframe['est_c2w'].to(self.device)
                        })

                    self.mesher.get_mesh(mesh_out_file, keyframe_dict_on_gpu, self.device)

    def final_run(self):

            with self.video.get_lock():
                self.video.map_counter.value += 1
                self.N = self.video.map_counter.value
                keyframe_ids = self.video.timestamp[:self.N]
                current_map_id = int(keyframe_ids[-1])
                batch = self.dataset[current_map_id]
                poses = self.video.get_pose(self.N, self.device)
                cur_c2w = poses[-1]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, ...]
                else:
                    batch[k] = torch.tensor([v])
            self.global_BA(batch, poses)
            self.mapping_idx[0] = current_map_id

            self.video.keyframe.add_keyframe(batch, self.video.map_counter.value)

            idx = int(self.mapping_idx[0].item())

            self.slam.save_imgs(idx, batch['depth'][0], batch['rgb'][0], cur_c2w)

            self.keyframe_dict.append(
                {'color': batch['rgb'][0].cpu(), 'depth': batch['depth'][0].cpu(),
                 'est_c2w': cur_c2w.clone().cpu()})

            pose_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
            np.save(
                f'{pose_save_path}/key_est_poses.npy',
                poses.cpu().numpy(),  # c2ws
            )









