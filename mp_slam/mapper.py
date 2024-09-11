# Use dataset object

import torch
import time
import os
import random
from optimization.utils import homogeneous_matrix_to_pose
import torch.nn.functional as F
class Mapper():
    def __init__(self, config, SLAM) -> None:
        self.config = config
        self.slam = SLAM
        self.model = SLAM.model
        self.tracking_idx = SLAM.tracking_idx
        self.mapping_idx = SLAM.mapping_idx
        self.mapping_first_frame = SLAM.mapping_first_frame
        self.keyframe = SLAM.keyframeDatabase
        self.map_optimizer = SLAM.map_optimizer
        self.device = SLAM.device
        self.dataset = SLAM.dataset
        self.video = SLAM.video
        # self.keyframe_dict = []

        # self.mesher = SLAM.mesher

        # self.est_c2w_data = SLAM.est_c2w_data
        #self.est_c2w_data = SLAM.video.poses

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

        # self.est_c2w_data[0] = c2w

        #self.est_c2w_data[0] = homogeneous_matrix_to_pose(c2w)

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
        self.keyframe.add_keyframe(batch, 1, filter_depth=self.config['mapping']['filter_depth'])
        # if self.config['mapping']['first_mesh']:
        #     self.slam.save_mesh(0)
        
        print('First frame mapping done')
        self.mapping_first_frame[0] = 1

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

        ###########for test
        # for ids in frame_ids_all:
        #     id = int(ids)
        #     poses = torch.stack((self.dataset[id]['c2w'].squeeze(), self.dataset[id]['c2w'].squeeze()), dim=0)

        # all the KF poses: 0, 5, 10, ...
        #poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        # frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))


        #fix pose0 /??self.est_c2w_data[0]

        poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)

        # print(poses_fixed)

        # cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(poses[1:])
        # pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
        # poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        # if pose_optimizer is not None:
        #     pose_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframe.sample_global_rays(self.config['mapping']['sample'])

##################?ids
            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W), max(self.config['mapping']['sample'] // len(self.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat([ids, -torch.ones((len(idx_cur)))]).to(torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            # rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            # rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            # rays_d = rays_d.reshape(-1, 3)

            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses[ids_all, None, :3, :3], -1)
            rays_o = poses[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            # loss = self.slam.get_loss_from_ret(ret, smooth=True)
            loss = self.slam.get_loss_from_ret(ret)
            print('loss', loss)
            loss.backward(retain_graph=True)
            
            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

####################################terminate
            # if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
            #     pose_optimizer.step()
            #     # get SE3 poses to do forward pass
            #     pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans)
            #     pose_optim = pose_optim.to(self.device)
            #     # So current pose is always unchanged

                # if self.config['mapping']['optim_cur']:
                #     poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                # else:
                #     #current_pose = self.est_c2w_data[cur_frame_id][None,...]
                #     current_pose = self.est_c2w_data[-1][None, ...]
                #     # SE3 poses

                # poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
#
#                 poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
#
#                 # zero_grad here
#                 pose_optimizer.zero_grad()
#
#         if pose_optimizer is not None and len(frame_ids_all) > 1:
#             for i in range(len(frame_ids_all[1:])):
#                 self.est_c2w_data[i] = self.slam.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
# ###################直接换video.pose
#             if self.config['mapping']['optim_cur']:
#                 print('Update current pose')
#                 #self.est_c2w_data[cur_frame_id] = self.slam.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
#                 self.est_c2w_data[-1] = self.slam.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
# #########################back?
            # for i in range(len(frame_ids_all[1:])):
            #     self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.slam.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
            #
            # if self.config['mapping']['optim_cur']:
            #     print('Update current pose')
            #     self.est_c2w_data[cur_frame_id] = self.slam.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

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
    def run(self):

        # Start mapping
        # while self.tracking_idx[0] < len(self.dataset)-1:
        if self.video.map_counter.value == 0 :
            batch = self.dataset[0]
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            time.sleep(0.1)
            # print(self.video.timestamp)
            # print('batchc2w', batch['c2w'])
            # test = torch.eye(4)
            self.slam.save_imgs(0, batch['depth'], batch['rgb'], batch['c2w'])

        else:

            # self.video.map_counter.value map了的帧数
            # self.video.counter.value track的keyframe的数量
            # 但是这时候track仍在进行？

            #################keyframe_ids一点点加上去优化，只要比self.video.counter.value小就继续
            while self.video.counter.value <= self.config['tracking']['warmup']:
                time.sleep(0.1)

            # print('111111111111map111111111111', self.video.map_counter.value)

            self.video.map_counter.value += 1

            self.N = self.video.map_counter.value
            keyframe_ids = self.video.timestamp[:self.N]
            # print('mapself.video.rm_timestamp', self.video.rm_timestamp)
            # keyframe_ids / self.video.rm_timestamp
            rm_timestamp_replaced = torch.where(self.video.rm_timestamp == 0,
                                                torch.tensor(float('nan'), device=self.video.rm_timestamp.device), self.video.rm_timestamp)
            mask = torch.isin(self.video.rm_timestamp, keyframe_ids)
            valid_mask = ~torch.isnan(rm_timestamp_replaced)
            final_mask = mask & valid_mask
            indices = torch.where(final_mask)[0]
            # print('indices', indices)
            if indices.numel() > 0:
                self.keyframe.del_keyframe(indices)

            current_map_id = int(keyframe_ids[-1])
            # print('current_map_id', current_map_id)
            batch = self.dataset[current_map_id]
            poses = self.video.get_pose(self.N, self.device)
            cur_c2w = poses[-1]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, ...]
                else:
                    batch[k] = torch.tensor([v])
#looptime
            # if current_map_id == 1300:
            #     self.loop_for_mapalign(batch, )

            # self.loop_for_mapalign(batch)

            # current batch + all video poses(include current
            self.global_BA(batch, poses)
            self.mapping_idx[0] = current_map_id

            if self.config['enable_loop_detect']:
                if current_map_id == self.config['loop_kf1_id']:
                    loop_id = self.config['loop_kf2_id']
                    batch_loop = self.dataset[loop_id]
                    self.tracking_render(batch_loop, cur_c2w)

            # if self.mapping_idx[0] % self.config['mapping']['keyframe_every'] == 0:
            #     self.keyframe.add_keyframe(batch)

            # print("add_keyframe", batch['frame_id'])

            if current_map_id > 0:
                self.keyframe.add_keyframe(batch, self.video.map_counter.value)

            # visualize
            #if self.mapping_idx[0] % self.config['mapping']['vis'] == 0:

            idx = int(self.mapping_idx[0].item())

            self.slam.save_imgs(idx, batch['depth'][0], batch['rgb'][0], cur_c2w)
#############################change, batch['depth']
            # self.keyframe_dict.append({'color': batch['rgb'][0].to(self.device), 'depth': batch['depth'][0].to(self.device), 'est_c2w': cur_c2w.clone()})

            if self.video.map_counter.value % 50 == 0 and self.video.map_counter.value > 0:
                self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])
                # mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'])
                # mesh_directory = os.path.join(mesh_save_path, 'mesh')
                #
                # # Create the directory if it doesn't exist
                # if not os.path.exists(mesh_directory):
                #     os.makedirs(mesh_directory)
                #
                # mesh_out_file = os.path.join(mesh_directory, f'{idx:05d}_mesh.ply')
                # self.mesher.get_mesh(mesh_out_file, self.keyframe_dict, self.device)






