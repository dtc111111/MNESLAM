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
        # self.keyframe = SLAM.keyframeDatabase

        self.map_optimizer = SLAM.map_optimizer
        self.device = SLAM.device
        self.dataset = SLAM.dataset
        self.video = SLAM.video

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
            loss = self.slam.get_loss_from_ret(ret)

            print('loss00000000', loss)

            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.video.keyframe.add_keyframe(batch, 1, filter_depth=self.config['mapping']['filter_depth'])
        # if self.config['mapping']['first_mesh']:
        #     self.slam.save_mesh(0)
        
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

            loss = self.slam.get_loss_from_ret(ret, smooth=True)
            print('loss', loss)
            loss.backward(retain_graph=False)

            self.map_optimizer.step()
            self.map_optimizer.zero_grad()

    def loop_for_mapalign(self, batch):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose = torch.nn.parameter.Parameter(torch.eye(4)).to(self.device)

        cur_trans = torch.nn.parameter.Parameter(pose.unsqueeze(0)[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.slam.matrix_to_tensor(pose.unsqueeze(0)[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['mapping']['lr_rot']},
                                           {"params": cur_trans, "lr": self.config['mapping']['lr_trans']}])

        pose_optimizer.zero_grad()
        for i in range(self.config['mapping']['iters']):

            indice = self.slam.select_samples(self.slam.dataset.H, self.slam.dataset.W, self.config['mapping']['sample'])
            indice_h, indice_w = indice % (self.slam.dataset.H), indice // (self.slam.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = pose[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * pose[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o.to(self.device), rays_d.to(self.device), target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret, sdf=False, fs=False, smooth=False )

            print('lossloooooooooooop', loss)

            loss.backward(retain_graph=True)

            pose_optimizer.step()
            pose = self.slam.matrix_from_tensor(cur_rot, cur_trans)
            pose = pose.squeeze(0).to(self.device)
            pose_optimizer.zero_grad()
            print(pose)

    def tracking_render(self, batch, cur_c2w):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        indice = None
        best_sdf_loss = None
        thresh = 0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.slam.get_pose_param_optim(cur_c2w[None, ...], mapping=False)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.slam.matrix_from_tensor(cur_rot, cur_trans)

            # Note here we fix the sampled points for optimization
            if indice is None:
                indice = self.slam.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2,
                                                  self.config['tracking']['sample'])

                # Slicing
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret, sdf=False, fs=False)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.slam.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh += 1

            if thresh > self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()

        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            loop_pose = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            loop_pose = c2w_est.detach().clone()[0]
        print('loop_pose', loop_pose)

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])

    def run(self):

        # Start mapping
        # while self.tracking_idx[0] < len(self.dataset)-1:
        if self.video.map_counter.value == 0:
            batch = self.dataset[0]
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            time.sleep(0.1)

        else:
            # self.video.map_counter.value map了的帧数
            # self.video.counter.value track的keyframe的数量
            # 但是这时候track仍在进行？

            #################keyframe_ids一点点加上去优化，只要比self.video.counter.value小就继续
            while (self.video.counter.value <= self.config['tracking']['warmup'] or self.video.map_counter.value >= self.video.counter.value -2) and (self.slam.tracking_finished < 1) :
                time.sleep(0.1)

            with self.video.get_lock():
                self.video.map_counter.value += 1
                # if self.video.map_counter.value == self.video.counter.value:
                #     time.sleep(0.5)
                self.N = self.video.map_counter.value
                keyframe_ids = self.video.timestamp[:self.N]

                print('counter', self.video.counter.value)
                print('map_counter', self.video.map_counter.value)
                print('keyframe_ids', keyframe_ids)
                print('timestamp', self.video.timestamp)

                current_map_id = int(keyframe_ids[-1])
                print('map_id', current_map_id)
                batch = self.dataset[current_map_id]
                poses = self.video.get_pose(self.N, self.device)
                cur_c2w = poses[-1]
                print('poses', poses.shape)
                print('cur_c2w', cur_c2w)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, ...]
                else:
                    batch[k] = torch.tensor([v])
# looptime
            # if current_map_id == 1300:
            #     self.loop_for_mapalign(batch, )

            # self.loop_for_mapalign(batch)

            # current batch + all video poses(include current
            self.global_BA(batch, poses)
            self.mapping_idx[0] = current_map_id

            if self.config['enable_loop_detect']:
                if current_map_id == self.config['loop_kf2_id']:

                    #load ckpt
                    save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name_1'])
                    poses_all = np.load(f'{save_path}/key_est_poses.npy') #tuple1100,4,4

                    dict = torch.load(f'{save_path}/final_checkpoint1099.pt')
                    self.model_shared.load_state_dict(dict['model'])

                    for i in range(self.config['mapping']['loop_iters']):
                        self.map_optimizer.zero_grad()
                        print('loop', i)
                        # Sample rays with real frame ids
                        # rays [bs, 7]
                        # frame_ids [bs]
                        self.H, self.W = self.config['cam']['H'] // self.config['data']['downsample'], \
                                         self.config['cam']['W'] // self.config['data']['downsample']

                        self.fx, self.fy = self.config['cam']['fx'] // self.config['data']['downsample'], \
                                           self.config['cam']['fy'] // self.config['data']['downsample']
                        self.cx, self.cy = self.config['cam']['cx'] // self.config['data']['downsample'], \
                                           self.config['cam']['cy'] // self.config['data']['downsample']

                        bs = self.config['training']['n_samples']
                        indices = torch.randint(0, self.dataset.total_pixels, (bs,))
                        rays_d_cam = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy).reshape(-1, 3)[
                            indices].to(self.device)  # 生成形状为 [bs, 3] 的射线

                        #N pose Bs rays

                        N = 20
                        indices = torch.randint(0, len(poses_all), (N,))
                        selected_poses = poses_all[indices]

                        selected_poses = torch.stack(
                            [torch.tensor(pose, dtype=torch.float32) for pose in selected_poses]).to(
                            self.device)

                        # 相机坐标系下的射线方向 N,Bs,1,3 * N,1,3,3 = N,Bs,3
                        rays_d = torch.sum(rays_d_cam[None, :, None, :] * selected_poses[:, None, :3, :3], -1)

                        rays_o = selected_poses[..., None, :3, -1].repeat(1, rays_d.shape[1],1).reshape(-1, 3)
                        rays_d = rays_d.reshape(-1, 3)
                        #torch.Size([92160, 3])

                        agent1_rendered_ret = self.model_shared.render_rays(rays_o, rays_d, target_d=None, loop=True)
                        agent1_rendered_rgb, agent1_rendered_depth = agent1_rendered_ret['rgb'], agent1_rendered_ret['depth']
                        # [bs,3], [bs,1]

                        agent1_rendered_depth = agent1_rendered_depth.unsqueeze(1)

                        ret = self.model.forward(rays_o, rays_d, agent1_rendered_rgb, agent1_rendered_depth)
                        # ret = {'rgb' : rgb_map, 'depth' :depth_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_var':depth_var,}
                        loss = self.slam.get_loss_from_ret(ret)

                        loss.backward()
                        self.map_optimizer.step()

                    idx = int(self.mapping_idx[0].item())
                    self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])
                    # loop_id = self.config['loop_kf2_id']
                    # batch_loop = self.dataset[loop_id]
                    # self.tracking_render(batch_loop, cur_c2w)


            self.video.keyframe.add_keyframe(batch, self.video.map_counter.value)

            idx = int(self.mapping_idx[0].item())

            self.slam.save_imgs(idx, batch['depth'][0], batch['rgb'][0], cur_c2w)

            pose_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
            np.save(
                f'{pose_save_path}/key_est_poses.npy',
                poses.cpu().numpy(),  # c2ws
            )

            if self.video.map_counter.value % 50 == 0 and self.video.map_counter.value > 0:
                self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])


    def final_run(self):

            with self.video.get_lock():
                self.video.map_counter.value += 1
                self.N = self.video.map_counter.value
                keyframe_ids = self.video.timestamp[:self.N]
                current_map_id = int(keyframe_ids[-1])
                print('map_id', current_map_id)
                batch = self.dataset[current_map_id]
                poses = self.video.get_pose(self.N, self.device)
                cur_c2w = poses[-1]
                print('poses', poses.shape)
                print('cur_c2w', cur_c2w)

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

            pose_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
            np.save(
                f'{pose_save_path}/key_est_poses.npy',
                poses.cpu().numpy(),  # c2ws
            )

            if self.video.map_counter.value % 50 == 0 and self.video.map_counter.value > 0:
                self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])






