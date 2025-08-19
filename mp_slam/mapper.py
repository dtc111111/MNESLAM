import numpy as np
from mp_slam.loop_detector import LoopDetector
import torch
import time
import os
import random
import torch.nn.functional as F
from datasets.utils import get_camera_rays
from optimization.utils import slerp_torch, matrix_to_quaternion_torch, quaternion_to_matrix_torch

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
        self.keyframe_dict_lock = SLAM.keyframe_dict_lock
        # If loop detection is enabled, initialize LoopDetector
        self.loop_detector = LoopDetector(config, SLAM) if config['enable_loop_detect'] else None
        self.c2w_matrices = []
        self.aligned_poses_c2w = None
        self.is_globalba = False

        # For convenience, copy camera parameters to self
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = self.dataset.H, self.dataset.W, self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy

        # For multi-agent collaboration
        # Ensure directions are initialized for collaborative mapping
        if self.dataset.rays_d is None:
            self.dataset.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        self.rank = SLAM.rank
        self.world_size = SLAM.world_size
        self.processed_keyframe_indices = set()
        self.fused_agents = set() # Tracks agents that this agent has already fused with.
        self.fused_frame_ids = set()
        self.all_agent_bounds = SLAM.all_agent_bounds
        self.db_lock = SLAM.descriptor_db_lock
        self.final_fusion_done = False
        self.use_bound_overlap = config.get('distillation', {}).get('use_bound_overlap', False)  # Retrieve the setting

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
        print(f"Agent {self.rank}: Starting first frame mapping...")
        if batch['frame_id'] != 0:
            raise ValueError('First frame mapping must be the first frame!')
        c2w = batch['c2w'].to(self.device)

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.slam.select_samples(self.slam.dataset.H, self.slam.dataset.W, self.config['mapping']['sample'])
            indice_h = indice % self.slam.dataset.H
            indice_w = torch.div(indice, self.slam.dataset.H, rounding_mode='trunc')
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
        with self.keyframe_dict_lock:
            self.keyframe_dict.append(
                {'color': batch['rgb'].cpu(),
                 'depth': batch['depth'].cpu(),
                 'agent_rank': self.rank,
                 'frame_id': batch['frame_id'],
                 'est_c2w': c2w.clone().cpu()})

        # Calculate and add the descriptor to the database for the first keyframe
        if self.config['enable_loop_detect']:
            self.loop_detector.detect_and_add(
                current_kf_id=batch['frame_id'],
                current_agent_id=self.rank,
                frame_rgb=batch['rgb']
            )

        print('First frame mapping done')
        self.mapping_first_frame[0] = 1
        self.slam.save_imgs(0, batch['depth'], batch['rgb'], batch['c2w'])
        self.slam.save_latest_checkpoint()
        self.video.map_counter.value += 1
        self.save_keyframe_data_atomic()
        self.slam.save_mesh(f'agent_{self.rank}', 0, voxel_size=self.config['mesh']['voxel_eval'])

    def mapping_optimize(self, batch, poses):
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
            rays, ids = self.video.keyframe.sample_global_rays(self.config['mapping']['sample'])

            idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W),
                                    max(self.config['mapping']['sample'] // len(self.video.keyframe.frame_ids),
                                        self.config['mapping']['min_pixels_cur']))

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
            loss.backward()

            self.map_optimizer.step()
            self.map_optimizer.zero_grad()

    def run(self):
        # Start mapping
        # while self.tracking_idx[0] < len(self.dataset)-1:
        if self.video.map_counter.value == 0:
            batch = self.dataset[0]
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            time.sleep(0.1)

        else:
            while (self.video.counter.value <= self.config['tracking'][
                'warmup'] or self.video.map_counter.value >= self.video.counter.value - 1) and (
                    self.slam.tracking_finished < 1):
                time.sleep(0.1)

            with self.video.get_lock():
                self.video.map_counter.value += 1
                self.N = self.video.map_counter.value
                keyframe_ids = self.video.timestamp[:self.N]
                current_map_id = int(keyframe_ids[-1])
                print('current_map_id', current_map_id)
                batch = self.dataset[current_map_id]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, ...]
                else:
                    batch[k] = torch.tensor([v])

            # Determine which poses to use for this mapping iteration
            if self.aligned_poses_c2w is not None:
                # If a loop closure has occurred, use the globally aligned poses.
                poses = self.aligned_poses_c2w[:self.N]
            else:
                # Otherwise, get the current poses from the video buffer.
                poses = self.video.get_pose(self.N, self.device)

            cur_c2w = poses[-1]
            # Call the new collaborative optimization function
            self.mapping_optimize(batch, poses)

            self.video.keyframe.add_keyframe(batch, self.video.map_counter.value)
            idx = int(self.mapping_idx[0].item())

            with self.keyframe_dict_lock:
                self.keyframe_dict.append(
                    {'color': batch['rgb'][0].cpu(),
                     'depth': batch['depth'][0].cpu(),
                     'agent_rank': self.rank,
                     'frame_id': idx,
                     'est_c2w': cur_c2w.clone().cpu()}) # Use the initial pose

            self.mapping_idx[0] = current_map_id

            # Save visualization with current pose
            idx = int(self.mapping_idx[0].item())
            self.slam.save_imgs(idx, batch['depth'][0], batch['rgb'][0], cur_c2w)

            self.save_keyframe_data_atomic()

            # Periodically save the latest model state for other agents to use in loop closure.
            self.slam.save_latest_checkpoint()

            if self.config['enable_loop_detect']:
                # Perform loop detection with NetVLAD and add the current frame to the database.
                print(f"Agent {self.rank}: Performing loop detection for keyframe {current_map_id}...")
                loop_closure_info = self.loop_detector.detect_and_add(
                    current_kf_id=current_map_id,
                    current_agent_id=self.rank,
                    frame_rgb=batch['rgb'][0] # Pass the [H, W, 3] RGB tensor
                )

                if loop_closure_info:
                    self.handle_loop_closure(loop_closure_info, current_map_id, cur_c2w)

            if (self.video.map_counter.value+1) % self.config['mapping']['mapping_save_stride'] == 0 and self.video.map_counter.value > 0:
                # Periodically save the latest model state for other agents to use in loop closure.
                # self.slam.save_latest_checkpoint()
                mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
                mesh_directory = os.path.join(mesh_save_path, 'mesh')
                # Create the directory if it doesn't exist
                if not os.path.exists(mesh_directory):
                    os.makedirs(mesh_directory)
                if self.config['is_co_sdf']:
                    self.slam.save_mesh(f'agent_{self.rank}', idx, voxel_size=self.config['mesh']['voxel_eval'])
                else:
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
        if not self.final_fusion_done:
            self.final_fusion_done = True
            # Perform bound-based fusion if enabled
            self.bound_based_fusion()

        with self.video.get_lock():
            self.video.map_counter.value += 1
            self.N = self.video.map_counter.value

            keyframe_ids = self.video.timestamp[:self.N]
            print(keyframe_ids)
            current_map_id = int(keyframe_ids[-1])
            batch = self.dataset[current_map_id]

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[None, ...]
            else:
                batch[k] = torch.tensor([v])

        if self.aligned_poses_c2w is not None:
            poses = self.aligned_poses_c2w[:self.N]
        else:
            poses = self.video.get_pose(self.N, self.device)
        
        cur_c2w = poses[-1]
        self.mapping_optimize(batch, poses)
        self.mapping_idx[0] = current_map_id

        self.video.keyframe.add_keyframe(batch, self.video.map_counter.value)

        idx = int(self.mapping_idx[0].item())

        self.slam.save_imgs(idx, batch['depth'][0], batch['rgb'][0], cur_c2w)
        self.slam.save_latest_checkpoint()
        
        # Add agent rank and frame_id to the keyframe data
        with self.keyframe_dict_lock:
            self.keyframe_dict.append(
                {'color': batch['rgb'][0].cpu(),
                 'depth': batch['depth'][0].cpu(),
                 'agent_rank': self.rank,
                 'frame_id': idx,
                 'est_c2w': cur_c2w.clone().cpu()})

        if self.config['enable_loop_detect']:
            print(f"Agent {self.rank}: Adding descriptor for first keyframe {batch['frame_id']}...")
            self.loop_detector.detect_and_add(
                current_kf_id=batch['frame_id'],
                current_agent_id=self.rank,
                frame_rgb=batch['rgb'][0]
            )

        self.save_keyframe_data_atomic()

    def handle_loop_closure(self, loop_closure_info, current_map_id, cur_c2w):
        """
        Handle a detected loop closure: 
        load foreign agent model, optimize relative pose, and apply alignment if this agent is the target.
        """
        other_agent_rank = loop_closure_info['match_agent_id']
        other_agent_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{other_agent_rank}')
        print(f"Agent {self.rank} found loop with Agent {other_agent_rank} (KF {loop_closure_info['match_kf_id']}) with similarity {loop_closure_info.get('similarity', 0):.4f}")

        # Track fused agents to avoid redundant fusion
        if other_agent_rank in self.fused_agents:
            print(f"Agent {self.rank} Loop to Agent {other_agent_rank} which already fused.")
        else:
            self.fused_agents.add(other_agent_rank)

        loop_id = (other_agent_rank, current_map_id)
        if loop_id in self.fused_frame_ids:
            return

        self.fused_frame_ids.add(loop_id)

        # Determine base and target agents and their initial poses
        match_kf_id = loop_closure_info['match_kf_id']
        if self.rank < other_agent_rank:
            base_agent_rank = self.rank
            target_agent_rank = other_agent_rank
            base_c2w = cur_c2w.clone().to(self.device)

            target_poses = np.load(os.path.join(other_agent_path, 'key_est_poses.npy'))
            target_timestamps = np.load(os.path.join(other_agent_path, 'key_timestamps.npy'))
            target_idx = np.where(target_timestamps == match_kf_id)[0][0]
            target_c2w_initial = torch.from_numpy(target_poses[target_idx]).to(self.device)
        else:
            base_agent_rank = other_agent_rank
            target_agent_rank = self.rank
            target_c2w_initial = cur_c2w.clone().to(self.device)

            base_poses = np.load(os.path.join(other_agent_path, 'key_est_poses.npy'))
            base_timestamps = np.load(os.path.join(other_agent_path, 'key_timestamps.npy'))
            base_idx = np.where(base_timestamps == match_kf_id)[0][0]
            base_c2w = torch.from_numpy(base_poses[base_idx]).to(self.device)

        print(f"Loop detected. Base agent: {base_agent_rank}, Target agent: {target_agent_rank}.")

        # Load foreign agent model into model_shared
        self.load_foreign_model(other_agent_rank)

        # Prepare optimizer for target pose
        target_rot, target_trans, pose_optimizer = self.slam.get_pose_param_optim(target_c2w_initial[None, ...], mapping=False)

        # Select models for base and target to avoid repeated rank checks
        model_for_base = self.model if base_agent_rank == self.rank else self.model_shared
        model_for_target = self.model if target_agent_rank == self.rank else self.model_shared

        # Prepare sampled rays and teacher targets (from base)
        with torch.no_grad():
            sample_size = self.config['mapping']['sample']
            rays_d_cam = self.dataset.rays_d.reshape(-1, 3)
            sample_indices = torch.randint(0, len(rays_d_cam), (sample_size,))
            rays_d_cam_batch = rays_d_cam[sample_indices].to(self.device)

            rays_o_base = base_c2w[:3, 3].unsqueeze(0).repeat(sample_size, 1)
            rays_d_base = torch.sum(rays_d_cam_batch[..., None, :] * base_c2w[:3, :3], dim=-1)

            base_rendered_ret = model_for_base.render_rays(rays_o_base, rays_d_base, target_d=None)
            target_rgb = base_rendered_ret['rgb'].detach()
            target_depth = base_rendered_ret['depth'].detach()

        best_loss = float('inf')
        best_target_c2w_est = target_c2w_initial.clone()

        # Optimization loop to align target to base
        for i in range(self.config['mapping']['loop_iters']):
            pose_optimizer.zero_grad()
            target_c2w_est = self.slam.matrix_from_tensor(target_rot, target_trans).squeeze(0)

            rays_o_target = target_c2w_est[:3, 3].unsqueeze(0).repeat(sample_size, 1)
            rays_d_target = torch.sum(rays_d_cam_batch[..., None, :] * target_c2w_est[:3, :3], dim=-1)

            rendered_ret = model_for_target.render_rays(rays_o_target, rays_d_target, target_d=None)
            rendered_rgb = rendered_ret['rgb']
            rendered_depth = rendered_ret['depth']

            loss_c = F.mse_loss(rendered_rgb, target_rgb)
            loss_d = F.mse_loss(rendered_depth, target_depth)
            loss = self.config['training']['rgb_weight'] * loss_c + self.config['training']['depth_weight'] * loss_d

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_target_c2w_est = target_c2w_est.detach().clone()

            loss.backward()
            pose_optimizer.step()
            if i % 10 == 0:
                print(f"Pose optim iter {i}: loss_c={loss_c.item():.6f} loss_d={loss_d.item():.6f} loss={loss.item():.6f}")

        # Compute relative transform that maps target -> base
        relative_transform = base_c2w @ torch.inverse(best_target_c2w_est)
        print(f"Agent {self.rank}: Calculated relative transform. Best loss: {best_loss:.6f}")

        # If this agent is the target, apply the transform to its trajectory/keyframes
        if target_agent_rank == self.rank:
            print(f"Agent {self.rank} is the target. Applying transformation to its own trajectory.")

            if self.aligned_poses_c2w is not None:
                current_agent_poses_c2w = self.aligned_poses_c2w
                num_keyframes = current_agent_poses_c2w.shape[0]
            else:
                with self.video.get_lock():
                    num_keyframes = self.video.counter.value
                    current_agent_poses_c2w = self.video.get_all_pose(self.device)[:num_keyframes]

            pre_alignment_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}', 'trajectory_debug')
            os.makedirs(pre_alignment_dir, exist_ok=True)
            pre_alignment_path = os.path.join(pre_alignment_dir, f'kf_{current_map_id:05d}_pre_alignment.npy')
            np.save(pre_alignment_path, current_agent_poses_c2w.cpu().numpy())
            print(f"Saved pre-alignment trajectory to {pre_alignment_path}")

            with self.video.get_lock():
                all_timestamps = self.video.timestamp[:num_keyframes].cpu().numpy()
            loop_kf_idx_arr = np.where(all_timestamps == current_map_id)[0]
            if len(loop_kf_idx_arr) > 0:
                loop_kf_idx = loop_kf_idx_arr[0]
            else:
                loop_kf_idx = num_keyframes - 1

            loop_kf_position = current_agent_poses_c2w[loop_kf_idx, :3, 3]
            all_positions = current_agent_poses_c2w[:, :3, 3]
            spatial_distances = torch.norm(all_positions - loop_kf_position, dim=1)

            loop_closure_config = self.config.get('loop_closure', {})
            decay_sigma = loop_closure_config.get('pose_decay_sigma', 5.0)
            min_weight = loop_closure_config.get('pose_decay_min_weight', 0.1)

            decay_weights = torch.exp(-spatial_distances.pow(2) / (2 * decay_sigma**2))
            final_weights = min_weight + (1.0 - min_weight) * decay_weights

            R_rel = relative_transform[:3, :3]
            t_rel = relative_transform[:3, 3]
            q_rel = matrix_to_quaternion_torch(R_rel)
            q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_rel.device, dtype=q_rel.dtype)

            interpolated_quaternions = slerp_torch(q_identity, q_rel, final_weights.unsqueeze(1))
            interpolated_rotations = quaternion_to_matrix_torch(interpolated_quaternions)

            interpolated_translations = final_weights.unsqueeze(1) * t_rel.unsqueeze(0)

            incremental_transforms = torch.eye(4, device=R_rel.device, dtype=R_rel.dtype).unsqueeze(0).repeat(num_keyframes, 1, 1)
            incremental_transforms[:, :3, :3] = interpolated_rotations
            incremental_transforms[:, :3, 3] = interpolated_translations

            transformed_poses_c2w = torch.matmul(incremental_transforms, current_agent_poses_c2w)

            post_alignment_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}', 'trajectory_debug')
            os.makedirs(post_alignment_dir, exist_ok=True)
            post_alignment_path = os.path.join(post_alignment_dir, f'kf_{current_map_id:05d}_post_alignment.npy')
            np.save(post_alignment_path, transformed_poses_c2w.cpu().numpy())
            print(f"Saved post-alignment trajectory to {post_alignment_path}")

            pose_diff = transformed_poses_c2w - current_agent_poses_c2w
            avg_diff = torch.norm(pose_diff, dim=(1,2)).mean().item()
            print(f"Avg per-pose c2w difference after transformation: {avg_diff:.6f}")

            self.aligned_poses_c2w = transformed_poses_c2w

            with self.keyframe_dict_lock:
                current_keyframe_list = list(self.keyframe_dict)
                for i in range(len(current_keyframe_list)):
                    kf_data = current_keyframe_list[i]
                    aligned_c2w = transformed_poses_c2w[i].cpu()
                    kf_data['est_c2w'] = aligned_c2w
                    self.keyframe_dict[i] = kf_data

            print(f"Agent {self.rank}: Applied transformation to its {num_keyframes} keyframes.")

    def compute_overlap_bound(self, bound1, bound2):
        """
        Computes the overlapping region between two bounding boxes.
        Args:
            bound1 (torch.Tensor): Bounding box of the first agent.
            bound2 (torch.Tensor): Bounding box of the second agent.
        Returns:
            torch.Tensor or None: Overlapping bounding box, or None if no overlap.
        """
        overlap = torch.empty_like(bound1)
        for i in range(3):  # Iterate over x, y, z dimensions
            overlap[i, 0] = torch.max(bound1[i, 0], bound2[i, 0])  # Max of mins
            overlap[i, 1] = torch.min(bound1[i, 1], bound2[i, 1])  # Min of maxs

        # Check for no overlap (min > max in any dimension)
        if torch.any(overlap[:, 0] > overlap[:, 1]):
            return None

        return overlap

    def get_keyframes_in_bound(self, agent_rank, bound):
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{agent_rank}')
        poses_path = os.path.join(agent_output_dir, 'key_est_poses.npy')
        timestamps_path = os.path.join(agent_output_dir, 'key_timestamps.npy')

        poses = np.load(poses_path)
        timestamps = np.load(timestamps_path)

        keyframes_in_bound = []
        for i, pose in enumerate(poses):
            position =pose [:3, 3]
            if (position[0] >= bound[0, 0] and position[0] <= bound[0, 1] and
                position[1] >= bound[1, 0] and position[1] <= bound[1, 1] and
                position[2] >= bound[2, 0] and position[2] <= bound[2, 1]):
                keyframes_in_bound.append({'kf_id': int(timestamps[i]), 'pose': torch.from_numpy(pose)})
        return keyframes_in_bound

    def find_multiple_matches(self, local_descriptors, foreign_descriptors, mutual_best=True):
        if not local_descriptors or not foreign_descriptors:
            return None

        local_des_tensor = torch.cat([item['descriptor'] for item in local_descriptors], dim=0).to(self.device)
        foreign_des_tensor = torch.cat([item['descriptor'] for item in foreign_descriptors], dim=0).to(self.device)
        sim_matrix = F.cosine_similarity(local_des_tensor.unsqueeze(1), foreign_des_tensor.unsqueeze(0), dim=2)

        best_foreign_for_local_val, best_foreign_for_local_idx = sim_matrix.max(dim=1)
        best_local_for_foreign_val, best_local_for_foreign_idx = sim_matrix.max(dim=0)

        matches = []
        for i in range(len(local_descriptors)):
            if best_foreign_for_local_val[i] < self.loop_detector.sim_threshold:
                continue

            match_idx_in_foreign = best_foreign_for_local_idx[i]
            if best_local_for_foreign_idx[match_idx_in_foreign] == i:
                matches.append({
                    'local_kf_id': local_descriptors[i]['kf_id'],
                    'foreign_kf_id': foreign_descriptors[match_idx_in_foreign]['kf_id'],
                    'similarity': best_foreign_for_local_val[i].item()
                })
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

    def get_pose_from_disk(self, agent_rank, kf_id):
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{agent_rank}')
        poses_path = os.path.join(agent_output_dir, 'key_est_poses.npy')
        timestamps_path = os.path.join(agent_output_dir, 'key_timestamps.npy')

        poses = np.load(poses_path)
        timestamps = np.load(timestamps_path)
        
        match_idx = np.where(timestamps == kf_id)[0]
        if len(match_idx) > 0:
            return torch.from_numpy(poses[match_idx[0]]).to(self.device)

    def save_keyframe_data_atomic(self):
        """
        Atomically saves the current keyframe poses and timestamps to disk.
        This ensures that other agents can safely read these files for loop closure.
        """
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
        os.makedirs(agent_output_dir, exist_ok=True)

        poses_path = os.path.join(agent_output_dir, 'key_est_poses.npy')
        timestamps_path = os.path.join(agent_output_dir, 'key_timestamps.npy')
        
        temp_poses_path = os.path.join(agent_output_dir, 'key_est_poses_tmp.npy')
        temp_timestamps_path = os.path.join(agent_output_dir, 'key_timestamps_tmp.npy')

        with self.video.get_lock():
            num_keyframes = self.video.counter.value
            keyframe_timestamps = self.video.timestamp[:num_keyframes].cpu().numpy()
            if self.aligned_poses_c2w is not None:
                keyframe_poses = self.aligned_poses_c2w[:num_keyframes].cpu().numpy()
                print(f"[Mapper {self.rank}] Using previously aligned poses for mapping.")
            else:
                keyframe_poses = self.video.get_all_pose(self.device)[:num_keyframes].cpu().numpy()

        np.save(temp_poses_path, keyframe_poses)
        os.replace(temp_poses_path, poses_path)

        np.save(temp_timestamps_path, keyframe_timestamps)
        os.replace(temp_timestamps_path, timestamps_path)

    def distillation(self, other_rank, expanded_foreign_kfs_for_distill, num_expanded_kfs):
        """
        Perform joint distillation from a foreign agent's keyframes into this agent's model_shared -> model.
        """
        for i in range(self.config['mapping']['distill_iters']):
            all_rays_o = []
            all_rays_d = []
            all_target_rgb = []
            all_target_depth = []

            sample_per_match = max(self.config['mapping']['sample'] // num_expanded_kfs,
                                   self.config['mapping']['min_pixels_cur']) if num_expanded_kfs > 0 else self.config['mapping']['sample']

            for kf_data in expanded_foreign_kfs_for_distill:
                foreign_pose = kf_data['pose'].to(self.device)

                rays_d_cam = self.dataset.rays_d.reshape(-1, 3)
                sample_indices = torch.randint(0, len(rays_d_cam), (sample_per_match,))
                rays_d_cam_batch = rays_d_cam[sample_indices].to(self.device)

                rays_o = foreign_pose[:3, 3].unsqueeze(0).repeat(sample_per_match, 1)
                rays_d = torch.sum(rays_d_cam_batch[..., None, :] * foreign_pose[:3, :3], dim=-1)

                all_rays_o.append(rays_o)
                all_rays_d.append(rays_d)

                with torch.no_grad():
                    teacher_ret = self.model_shared.render_rays(rays_o, rays_d, target_d=None)
                    all_target_rgb.append(teacher_ret['rgb'].detach())
                    all_target_depth.append(teacher_ret['depth'].detach().unsqueeze(-1))

            if not all_rays_o:
                print(f"Skipping.")
                continue

            all_rays_o = torch.cat(all_rays_o, dim=0)
            all_rays_d = torch.cat(all_rays_d, dim=0)
            all_target_rgb = torch.cat(all_target_rgb, dim=0)
            all_target_depth = torch.cat(all_target_depth, dim=0)

            self.map_optimizer.zero_grad()

            student_ret = self.model.forward(all_rays_o, all_rays_d, all_target_rgb, all_target_depth)
            distillation_loss = self.slam.get_loss_from_ret(student_ret, is_co_sdf=self.config['is_co_sdf'])

            distillation_loss.backward()
            self.map_optimizer.step()

        print(f"Agent {self.rank}: Joint distillation with Agent {other_rank} complete.")
        # Save the final fused mesh
        self.slam.save_mesh(f'agent_{self.rank}', f'final_fused_with_{other_rank}', voxel_size=self.config['mesh']['voxel_final'])

    def bound_based_fusion(self):
        """
        Perform bound-based fusion across fused agents using overlapping bounds.
        """
        if not self.use_bound_overlap or self.world_size <= 1:
            return

        for other_rank in self.fused_agents:
            current_agent_bound = self.all_agent_bounds[self.rank]
            other_agent_bound = self.all_agent_bounds.get(other_rank)
            if other_agent_bound is None:
                continue

            overlap_bound = self.compute_overlap_bound(current_agent_bound, other_agent_bound)
            if overlap_bound is None:
                continue

            local_kfs_in_overlap = self.get_keyframes_in_bound(self.rank, overlap_bound)
            foreign_kfs_in_overlap = self.get_keyframes_in_bound(other_rank, overlap_bound)
            if not local_kfs_in_overlap or not foreign_kfs_in_overlap:
                continue

            local_kf_ids = {kf['kf_id'] for kf in local_kfs_in_overlap}
            foreign_kf_ids = {kf['kf_id'] for kf in foreign_kfs_in_overlap}

            with self.db_lock:
                db_copy = list(self.slam.descriptor_db)

            local_descriptors = [item for item in db_copy if item['agent_id'] == self.rank and item['kf_id'] in local_kf_ids]
            foreign_descriptors = [item for item in db_copy if item['agent_id'] == other_rank and item['kf_id'] in foreign_kf_ids]

            matches = self.find_multiple_matches(local_descriptors, foreign_descriptors)
            min_matches = self.config.get('distillation', {}).get('min_matches_for_fusion', 3)
            if not matches or len(matches) <= min_matches:
                continue

            foreign_kf_ids_from_matches = [m['foreign_kf_id'] for m in matches]
            min_foreign_id = min(foreign_kf_ids_from_matches)
            max_foreign_id = max(foreign_kf_ids_from_matches)

            expanded_foreign_kfs_for_distill = [
                kf for kf in foreign_kfs_in_overlap
                if min_foreign_id <= kf['kf_id'] <= max_foreign_id
            ]
            num_expanded_kfs = len(expanded_foreign_kfs_for_distill)

            pose_pairs = []
            for match in matches:
                local_pose = self.get_pose_from_disk(self.rank, match['local_kf_id'])
                foreign_pose = self.get_pose_from_disk(other_rank, match['foreign_kf_id'])
                if local_pose is not None and foreign_pose is not None:
                    pose_pairs.append((local_pose, foreign_pose))

            if len(pose_pairs) < min_matches:
                continue

            # Load foreign model and set up model_shared
            self.load_foreign_model(other_rank)

            # Delegate to joint distillation
            self.distillation(other_rank, expanded_foreign_kfs_for_distill, num_expanded_kfs)

    def load_foreign_model(self, other_rank):
        """Load checkpoint for other agent and configure self.model_shared accordingly.
        Returns the loaded checkpoint dict.
        """
        other_agent_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{other_rank}')
        other_agent_checkpoint_path = os.path.join(other_agent_path, 'latest_checkpoint.pt')
        other_agent_checkpoint = torch.load(other_agent_checkpoint_path, map_location=self.device)
        # Load model weights
        self.model_shared.load_state_dict(other_agent_checkpoint['model'])
        # Copy auxiliary state if present
        if 'all_planes' in other_agent_checkpoint:
            self.model_shared.all_planes = other_agent_checkpoint['all_planes']
        # Set boundary context if present
        if 'bound' in other_agent_checkpoint:
            self.model_shared.bound = other_agent_checkpoint['bound'].to(self.device)
        if 'bounding_box' in other_agent_checkpoint:
            self.model_shared.bounding_box = other_agent_checkpoint['bounding_box'].to(self.device)
        self.model_shared.eval()
        return other_agent_checkpoint
