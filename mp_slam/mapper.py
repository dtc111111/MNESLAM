# Use dataset object
import numpy as np
from mp_slam.loop_detector import LoopDetector
from tqdm import tqdm
import torch
import time
import os
import random
from lietorch import SE3
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from optimization.utils import homogeneous_matrix_to_pose
import torch.nn.functional as F
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion, cam_pose_to_matrix, homogeneous_matrix_to_pose
from datasets.utils import get_camera_rays
import lietorch
from optimization.utils import slerp_torch, matrix_to_quaternion_torch, quaternion_to_matrix_torch
from tools.coslam_eval.cull_mesh import cull_one_mesh
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
        print(f"[Mapper {self.rank}] Using bound overlap for distillation: {self.use_bound_overlap}")

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
            indice_h, indice_w = indice % (self.slam.dataset.H), indice // (self.slam.dataset.H)
            rays_d_cam = batch['direction'][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o.to(self.device), rays_d.to(self.device), target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret, is_co_sdf=self.config['is_co_sdf'])

            print(f'agent{self.rank}', 'loss00000000', loss)

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
            print(f"Agent {self.rank}: Adding descriptor for last keyframe {batch['frame_id']}...")
            self.loop_detector.detect_and_add(
                current_kf_id=batch['frame_id'],
                current_agent_id=self.rank,
                frame_rgb=batch['rgb']
            )

        print('First frame mapping done')
        self.mapping_first_frame[0] = 1
        self.slam.save_imgs(0, batch['depth'], batch['rgb'], batch['c2w'])
        print('save_frame_0!')
        self.slam.save_latest_checkpoint()
        self.video.map_counter.value += 1
        self.save_keyframe_data_atomic()
        # mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
        # mesh_directory = os.path.join(mesh_save_path, 'mesh')
        # # Create the directory if it doesn't exist
        # if not os.path.exists(mesh_directory):
        #     os.makedirs(mesh_directory)

        # mesh_out_file = os.path.join(mesh_directory, '00000_mesh.ply')

        # with self.keyframe_dict_lock:
        #     keyframe_dict_on_cpu = list(self.keyframe_dict)

        #     keyframe_dict_on_gpu = []
        #     for keyframe in keyframe_dict_on_cpu:
        #         keyframe_dict_on_gpu.append({
        #             'color': keyframe['color'].to(self.device),
        #             'depth': keyframe['depth'].to(self.device),
        #             'est_c2w': keyframe['est_c2w'].to(self.device)
        #         })
###################e
        # self.mesher.get_mesh(mesh_out_file, keyframe_dict_on_gpu, self.device)
##################co
        self.slam.save_mesh(f'agent_{self.rank}', 0, voxel_size=self.config['mesh']['voxel_eval'])

        # self.c2w_matrices.append(batch['c2w'].to(self.device))

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

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.video.keyframe.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            # idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W), max(self.config['mapping']['sample'] // len(self.video.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
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
            print(f'agent{self.rank}', 'loss', loss)
            loss.backward()

            self.map_optimizer.step()
            self.map_optimizer.zero_grad()

    def global_BA(self, batch):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        poses = self.c2w_matrices[:self.N]
        poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)

        cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(poses[1:])
        pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
        poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        pose_optimizer.zero_grad()
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.video.keyframe.sample_global_rays(self.config['mapping']['sample'])

            # TODO: Checkpoint...
            # idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W), max(self.config['mapping']['sample'] // len(self.video.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
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
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.slam.get_loss_from_ret(ret, is_co_sdf=self.config['is_co_sdf'])

            loss.backward(retain_graph=True)

            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:

                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                # zero_grad here
                pose_optimizer.zero_grad()

        for i in range(self.N-2): # pose0fixed 1-self.N-2 =  0-self.N-3
            self.c2w_matrices[int(self.video.keyframe.frame_ids[i + 1].item())] = \
            self.slam.matrix_from_tensor(cur_rot[i:i + 1], cur_trans[i:i + 1]).detach().clone()[0]
        self.c2w_matrices[self.N - 1] = self.slam.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

    def _visualize_model_output(self, c2w1, foreign_frame_id, title):
        """
        Visualizes the RGB and depth outputs of a given model within the overlapping region.
        """
        print(f"Agent {self.rank}: Visualizing {title} for frame {foreign_frame_id}...")
        H, W = self.H, self.W
        device = self.device

        with torch.no_grad():
            # Render the view from c2w1 using the provided model
            depth_map, rgb_map = self.model_shared.render_img(c2w1, device)
            rgb_np = np.clip(rgb_map.cpu().numpy(), 0, 1)
            depth_np = depth_map.cpu().numpy()

            depth_self, rgb_self = self.model.render_img(c2w1, device)
            rgb_self_np = np.clip(rgb_self.cpu().numpy(), 0, 1)
            depth_self_np = depth_self.cpu().numpy()

            # Plotting
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(
                f'Agent {self.rank} - {title} in (Foreign {foreign_frame_id}',
                fontsize=18,
            )

            axs[0, 0].imshow(rgb_np)
            axs[0, 0].set_title("Foreign Output (RGB)", fontsize=12)
            axs[0, 0].axis('off')

            axs[1, 0].imshow(depth_np, cmap='plasma')
            axs[1, 0].set_title("Model Output (Depth)", fontsize=12)
            axs[1, 0].axis('off')

            axs[0, 1].imshow(rgb_self_np)
            axs[0, 1].set_title("Self Output (RGB)", fontsize=12)
            axs[0, 1].axis('off')

            axs[1, 1].imshow(depth_self_np, cmap='plasma')
            axs[1, 1].set_title("Self Output (Depth)", fontsize=12)
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)

            # Save the figure
            save_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}', 'distill_vis')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{title.replace(" ", "_")}_foreign{foreign_frame_id:05d}.png')
            plt.savefig(save_path)
            plt.close(fig)

            print(f"Agent {self.rank}: Saved {title} visualization to {save_path}")

    def run(self):

        # Start mapping
        # while self.tracking_idx[0] < len(self.dataset)-1:
        if self.video.map_counter.value == 0:
            batch = self.dataset[0]
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            time.sleep(0.1)

        else:
            # self.video.map_counter.value 已映射的帧数
            # self.video.counter.value track的keyframe的数量
            # 但是这时候track可能仍在进行
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

                # cur_c2w = poses[-1]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[None, ...]
                else:
                    batch[k] = torch.tensor([v])

            # if self.slam.tracking_finished < 1 or self.video.map_counter.value < 3: #warmup change later
            #     poses = self.video.get_pose(self.N, self.device)
            #     cur_c2w = poses[-1]
            #     self.mapping_optimize(batch, poses)
            # else:
            #     if not self.is_globalba:
            #         print('global bundle adjustment!')
            #         self.c2w_matrices = self.video.get_all_pose(self.device)
            #         self.is_globalba = True

             #  self.global_BA(batch)
             #    cur_c2w = self.c2w_matrices[self.N - 1]
            # Determine which poses to use for this mapping iteration
            if self.aligned_poses_c2w is not None:
                # If a loop closure has occurred, use the globally aligned poses.
                poses = self.aligned_poses_c2w[:self.N]
                print(f"[Mapper {self.rank}] Using aligned pose #{self.N} for mapping.")
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

            if self.aligned_poses_c2w is not None:
                # If a loop closure has occurred, use the globally aligned poses.
                poses = self.aligned_poses_c2w[:self.N]
                print(f"[Mapper {self.rank}] Using aligned pose #{self.N} for mapping.")
            else:
                # Otherwise, get the current poses from the video buffer.
                poses = self.video.get_pose(self.N, self.device)
                
            cur_c2w = poses[-1]

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
                    other_agent_rank = loop_closure_info['match_agent_id']
                    other_agent_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{other_agent_rank}')
                    print(f"Agent {self.rank} found loop with Agent {other_agent_rank} (KF {loop_closure_info['match_kf_id']}) with similarity {loop_closure_info['similarity']:.4f}")

                    # Add the agent to the set of fused agents to prevent re-fusion.
                    # This makes loop detection continuous without redundant fusions.

                    if other_agent_rank in self.fused_agents:
                        print(f"[Agent {self.rank}] Loop to Agent {other_agent_rank} which already fused.")
                    else:
                        self.fused_agents.add(other_agent_rank)

                    loop_id = (other_agent_rank, current_map_id)
                    if loop_id in self.fused_frame_ids:
                        print(f"[Agent {self.rank}] Loop to Agent {other_agent_rank} @ Frame {current_map_id} already fused.")
                    else:
                        self.fused_frame_ids.add(loop_id)
                        # Determine the base agent (smaller rank) and target agent (larger rank).
                        if self.rank < other_agent_rank:
                            base_agent_rank = self.rank
                            target_agent_rank = other_agent_rank
                            # The current agent is the base, its pose is fixed.
                            base_c2w = cur_c2w.clone().to(self.device)
                            # Need to load the target agent's pose from disk to find the loop frame.
                            target_poses = np.load(os.path.join(other_agent_path, 'key_est_poses.npy'))
                            target_timestamps = np.load(os.path.join(other_agent_path, 'key_timestamps.npy'))
                            match_kf_id = loop_closure_info['match_kf_id']
                            try:
                                target_idx = np.where(target_timestamps == match_kf_id)[0][0]
                                target_c2w_initial = torch.from_numpy(target_poses[target_idx]).to(self.device)
                            except IndexError:
                                print(f"ERROR: Could not find KF {match_kf_id} in Agent {target_agent_rank}'s saved poses.")
                        else:
                            base_agent_rank = other_agent_rank
                            target_agent_rank = self.rank
                            target_c2w_initial = cur_c2w.clone().to(self.device)
                            base_poses = np.load(os.path.join(other_agent_path, 'key_est_poses.npy'))
                            base_timestamps = np.load(os.path.join(other_agent_path, 'key_timestamps.npy'))
                            match_kf_id = loop_closure_info['match_kf_id']
                            try:
                                base_idx = np.where(base_timestamps == match_kf_id)[0][0]
                                base_c2w = torch.from_numpy(base_poses[base_idx]).to(self.device)
                            except IndexError:
                                print(f"ERROR: Could not find KF {match_kf_id} in Agent {base_agent_rank}'s saved poses.")

                        print(f"Loop detected. Base agent: {base_agent_rank}, Target agent: {target_agent_rank}.")

                        other_agent_checkpoint_path = os.path.join(other_agent_path, 'latest_checkpoint.pt')
                        print(f"Agent {self.rank}: Loading other agent's model from {other_agent_checkpoint_path}")
                        other_agent_checkpoint = torch.load(other_agent_checkpoint_path, map_location=self.device)
                        self.model_shared.load_state_dict(other_agent_checkpoint['model'])
                        self.model_shared.all_planes = other_agent_checkpoint['all_planes']
                        
                        # Load the correct boundary context for model_shared.
                        foreign_bound = other_agent_checkpoint['bound'].to(self.device)
                        foreign_bounding_box = other_agent_checkpoint['bounding_box'].to(self.device)
                        original_bound = self.model_shared.bound
                        original_bounding_box = self.model_shared.bounding_box
                        self.model_shared.bound = foreign_bound
                        self.model_shared.bounding_box = foreign_bounding_box
                        self.model_shared.eval()

                        #  Visualize the loaded foreign model's view before optimization.
                        with torch.no_grad():
                            if self.rank < other_agent_rank:
                                foreign_pose_to_vis = target_c2w_initial
                                vis_title_prefix = f"Agent_{self.rank}_sees_Agent_{other_agent_rank}"
                            else:
                                foreign_pose_to_vis = base_c2w
                                vis_title_prefix = f"Agent_{self.rank}_sees_Agent_{base_agent_rank}"

                            print(f"Visualizing foreign model's view with pose:\n{foreign_pose_to_vis}")
                            foreign_depth_vis, foreign_rgb_vis = self.model_shared.render_img(foreign_pose_to_vis, self.device, gt_depth=None)
                            foreign_rgb_np = np.clip(foreign_rgb_vis.cpu().numpy(), 0, 1)
                            foreign_depth_np = foreign_depth_vis.cpu().numpy()

                            fig_vis, axs_vis = plt.subplots(1, 2, figsize=(12, 6))
                            fig_vis.suptitle(f"{vis_title_prefix} (at its loop KF {loop_closure_info['match_kf_id']})")
                            axs_vis[0].imshow(foreign_rgb_np)
                            axs_vis[0].set_title('Foreign Rendered RGB')
                            axs_vis[0].axis('off')
                            axs_vis[1].imshow(foreign_depth_np, cmap='plasma')
                            axs_vis[1].set_title('Foreign Rendered Depth')
                            axs_vis[1].axis('off')

                            vis_save_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}', 'foreign_view_vis')
                            os.makedirs(vis_save_dir, exist_ok=True)
                            vis_save_path = os.path.join(vis_save_dir, f'foreign_view_at_localKF_{current_map_id:05d}_from_foreignKF_{loop_closure_info["match_kf_id"]:05d}.png')
                            plt.savefig(vis_save_path)
                            plt.close(fig_vis)
                            print(f"Saved foreign view visualization to {vis_save_path}")
                        
                        print(f"Agent {self.rank}: Optimizing relative pose between its KF {current_map_id} and Agent {other_agent_rank}'s KF {loop_closure_info['match_kf_id']}")

                        # We optimize target_c2w to align its rendered output with that of base_c2w.
                        # Therefore, we create optimizable parameters for target_c2w.
                        target_rot, target_trans, pose_optimizer = self.slam.get_pose_param_optim(target_c2w_initial[None, ...], mapping=False)

                        with torch.no_grad():
                            sample_size = self.config['mapping']['sample']  # e.g., 512
                            H, W = self.config['cam']['H'], self.config['cam']['W']
                            rays_d_cam = self.dataset.rays_d.reshape(-1, 3)
                            sample_indices = torch.randint(0, len(rays_d_cam), (sample_size,))
                            rays_d_cam_batch = rays_d_cam[sample_indices].to(self.device)

                            # Get rays_o, rays_d from base_c2w
                            rays_o_base = base_c2w[:3, 3].unsqueeze(0).repeat(sample_size, 1)
                            rays_d_base = torch.sum(rays_d_cam_batch[..., None, :] * base_c2w[:3, :3], dim=-1)

                            # Render target RGB and depth from base_c2w as target
                            if self.rank == base_agent_rank:
                                base_rendered_ret = self.model.render_rays(rays_o_base, rays_d_base, target_d=None)
                            else:
                                base_rendered_ret = self.model_shared.render_rays(rays_o_base, rays_d_base, target_d=None)
                            target_rgb = base_rendered_ret['rgb'].detach()
                            target_depth = base_rendered_ret['depth'].detach()


                        best_loss = float('inf')
                        best_target_c2w_est = target_c2w_initial.clone()

                        # Optimization loop
                        for i in range(self.config['mapping']['loop_iters']):
                            pose_optimizer.zero_grad()
                            target_c2w_est = self.slam.matrix_from_tensor(target_rot, target_trans).squeeze(0)

                            # Generate new rays with the current estimated pose (same directions).
                            rays_o_target = target_c2w_est[:3, 3].unsqueeze(0).repeat(sample_size, 1)
                            rays_d_target = torch.sum(rays_d_cam_batch[..., None, :] * target_c2w_est[:3, :3], dim=-1)

                            if self.rank == target_agent_rank:
                                # Target is self, use local model
                                rendered_ret = self.model.render_rays(rays_o_target, rays_d_target, target_d=None)
                            else:
                                # Target is other agent, use shared/foreign model
                                rendered_ret = self.model_shared.render_rays(rays_o_target, rays_d_target, target_d=None)
                            rendered_rgb = rendered_ret['rgb']
                            rendered_depth = rendered_ret['depth']

                            # Calculate consistency loss
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

                        # Calculate the final relative transform T_target_base = P_base * P_target_inv.
                        # This transform aligns the target agent's coordinate system to the base agent's.
                        relative_transform = base_c2w @ torch.inverse(best_target_c2w_est)
                        print(f"Agent {self.rank}: Calculated relative transform. Best loss: {best_loss:.6f}")

                        if self.rank == target_agent_rank:
                            print(f"\nAgent {self.rank} is the target. Applying transformation to its own trajectory.")
                            
                            # Get all poses of the current agent for transformation.
                            if self.aligned_poses_c2w is not None:
                                # If already aligned, apply the new transform on top of the existing aligned poses.
                                current_agent_poses_c2w = self.aligned_poses_c2w
                                num_keyframes = current_agent_poses_c2w.shape[0]
                                print(f"[Mapper {self.rank}] Applying new alignment on top of existing {num_keyframes} aligned poses.")
                            else:
                                # First alignment, get poses from the video buffer.
                                with self.video.get_lock():
                                    num_keyframes = self.video.counter.value
                                    current_agent_poses_c2w = self.video.get_all_pose(self.device)[:num_keyframes]

                            # 保存对齐前的轨迹
                            pre_alignment_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}', 'trajectory_debug')
                            os.makedirs(pre_alignment_dir, exist_ok=True)
                            pre_alignment_path = os.path.join(pre_alignment_dir, f'kf_{current_map_id:05d}_pre_alignment.npy')
                            np.save(pre_alignment_path, current_agent_poses_c2w.cpu().numpy())
                            print(f"Saved pre-alignment trajectory to {pre_alignment_path}")

                            if self.aligned_poses_c2w is None:
                                # 首次对齐：应用刚性变换，不加权
                                print(f"Agent {self.rank}: Applying first-time rigid transformation to all poses.")
                                # 应用变换: P_new = T_rel @ P_old
                                transformed_poses_c2w = torch.matmul(relative_transform[None, :, :], current_agent_poses_c2w)
                            else:
                                # 后续对齐：应用带权重衰减的变换
                                print(f"Agent {self.rank}: Applying weighted pose graph adjustment on already aligned poses.")

                                # 找到当前回环帧在位姿列表中的索引
                                with self.video.get_lock():
                                    all_timestamps = self.video.timestamp[:num_keyframes].cpu().numpy()
                                loop_kf_idx_arr = np.where(all_timestamps == current_map_id)[0]
                                
                                if len(loop_kf_idx_arr) > 0:
                                    loop_kf_idx = loop_kf_idx_arr[0]
                                else:
                                    # 如果找不到（理论上不应发生），则以最后一帧为锚点
                                    print(f"[WARN] Could not find loop KF ID {current_map_id} in timestamps. Using last frame as anchor.")
                                    loop_kf_idx = num_keyframes - 1

                                # 计算所有关键帧到回环帧的空间距离
                                loop_kf_position = current_agent_poses_c2w[loop_kf_idx, :3, 3]
                                all_positions = current_agent_poses_c2w[:, :3, 3]
                                spatial_distances = torch.norm(all_positions - loop_kf_position, dim=1)

                                # 根据距离计算权重 (高斯衰减)
                                loop_closure_config = self.config.get('loop_closure', {})
                                decay_sigma = loop_closure_config.get('pose_decay_sigma', 5.0) # 默认5米
                                min_weight = loop_closure_config.get('pose_decay_min_weight', 0.1) # 默认最小权重0.1
                                
                                decay_weights = torch.exp(-spatial_distances.pow(2) / (2 * decay_sigma**2))
                                final_weights = min_weight + (1.0 - min_weight) * decay_weights

                                # 分别对旋转和平移进行加权插值
                                print(f"Agent {self.rank}: Applying weighted pose graph adjustment via Quaternion SLERP.")
                                
                                # 分解相对变换并转换为四元数
                                R_rel = relative_transform[:3, :3]
                                t_rel = relative_transform[:3, 3]
                                q_rel = matrix_to_quaternion_torch(R_rel) # (w,x,y,z)
                                q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_rel.device, dtype=q_rel.dtype)

                                # 对旋转进行球面线性插值 (SLERP)
                                interpolated_quaternions = slerp_torch(q_identity, q_rel, final_weights.unsqueeze(1))
                                interpolated_rotations = quaternion_to_matrix_torch(interpolated_quaternions) # [N, 3, 3]

                                # 对平移进行线性插值 (LERP)
                                interpolated_translations = final_weights.unsqueeze(1) * t_rel.unsqueeze(0) # [N, 3]

                                # 重新组合成 N 个增量变换矩阵
                                incremental_transforms = torch.eye(4, device=R_rel.device, dtype=R_rel.dtype).unsqueeze(0).repeat(num_keyframes, 1, 1)
                                incremental_transforms[:, :3, :3] = interpolated_rotations
                                incremental_transforms[:, :3, 3] = interpolated_translations

                                # 应用变换: P_new = T_inc @ P_old
                                transformed_poses_c2w = torch.matmul(incremental_transforms, current_agent_poses_c2w)
                            
                            # 保存对齐后的轨迹
                            post_alignment_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}', 'trajectory_debug')
                            os.makedirs(post_alignment_dir, exist_ok=True) # 确保目录存在
                            post_alignment_path = os.path.join(post_alignment_dir, f'kf_{current_map_id:05d}_post_alignment.npy')
                            np.save(post_alignment_path, transformed_poses_c2w.cpu().numpy())
                            print(f"Saved post-alignment trajectory to {post_alignment_path}")

                            # 检查所有 pose 的平均变化
                            pose_diff = transformed_poses_c2w - current_agent_poses_c2w
                            avg_diff = torch.norm(pose_diff, dim=(1,2)).mean().item()
                            print(f"Avg per-pose c2w difference after transformation: {avg_diff:.6f}")

                            # 将对齐后的位姿存储在本地，不再更新共享的 video 缓冲区
                            self.aligned_poses_c2w = transformed_poses_c2w

                            # 更新本地 keyframe_dict
                            with self.keyframe_dict_lock:
                                current_keyframe_list = list(self.keyframe_dict)
                                # 用已经计算好的 transformed_poses_c2w 替换 keyframe_dict 中对应帧
                                for i in range(len(current_keyframe_list)):
                                    kf_data = current_keyframe_list[i]
                                    # 确保位姿索引匹配（第 i 个 keyframe 对应第 i 个 pose）
                                    aligned_c2w = transformed_poses_c2w[i].cpu()
                                    # 替换 est_c2w（也可以保留原始）
                                    kf_data['est_c2w'] = aligned_c2w
                                    # 更新回字典
                                    self.keyframe_dict[i] = kf_data

                            print(f"Agent {self.rank}: Applied transformation to its {num_keyframes} keyframes.")

            if (self.video.map_counter.value+1) % 10 == 0 and self.video.map_counter.value > 0:
                # Periodically save the latest model state for other agents to use in loop closure.
                # self.slam.save_latest_checkpoint()
                if self.config['is_co_sdf']:
                    mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
                    mesh_directory = os.path.join(mesh_save_path, 'mesh')

                    # Create the directory if it doesn't exist
                    if not os.path.exists(mesh_directory):
                        os.makedirs(mesh_directory)
                    self.slam.save_mesh(f'agent_{self.rank}', idx, voxel_size=self.config['mesh']['voxel_eval'])
                # else:
                #     print('save_eslam_mesh!')
                #     mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
                #     mesh_directory = os.path.join(mesh_save_path, 'mesh')

                #     # Create the directory if it doesn't exist
                #     if not os.path.exists(mesh_directory):
                #         os.makedirs(mesh_directory)

                #     mesh_out_file = os.path.join(mesh_directory, f'{idx:05d}_mesh.ply')

                #     keyframe_dict_on_cpu = self.keyframe_dict

                #     keyframe_dict_on_gpu = []
                #     for keyframe in keyframe_dict_on_cpu:
                #         keyframe_dict_on_gpu.append({
                #             'color': keyframe['color'].to(self.device),
                #             'depth': keyframe['depth'].to(self.device),
                #             'est_c2w': keyframe['est_c2w'].to(self.device)
                #         })

                #     self.mesher.get_mesh(mesh_out_file, keyframe_dict_on_gpu, self.device)

                #     suffix = "occlusion"
                #     cull_save_path = mesh_out_file.replace(".ply", "_cull_{}.ply".format(suffix))
                #     cull_one_mesh(self.config, poses.cpu().numpy(), mesh_out_file, cull_save_path, save_unseen=False,
                #         remove_occlusion=True, scene_bounds=None,
                #         eps=0.03, th_obs=0, silent=True, platform='egl')

    def compute_overlap_bound(self, bound1, bound2):
        """
        Computes the overlapping region between two bounding boxes.
        Args:
            bound1 (torch.Tensor): Bounding box of the first agent.
            bound2 (torch.Tensor): Bounding box of the second agent.
        Returns:
            torch.Tensor or None: Overlapping bounding box, or None if no overlap.
        """
        # 假设 bound 格式为 [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        overlap = torch.empty_like(bound1)
        for i in range(3):  # Iterate over x, y, z dimensions
            overlap[i, 0] = torch.max(bound1[i, 0], bound2[i, 0])  # Max of mins
            overlap[i, 1] = torch.min(bound1[i, 1], bound2[i, 1])  # Min of maxs

        # Check for no overlap (min > max in any dimension)
        if torch.any(overlap[:, 0] > overlap[:, 1]):
            return None

        return overlap

    def get_keyframes_in_bound(self, agent_rank, bound):
        """
        从磁盘加载关键帧数据，并根据边界框进行筛选。
        返回一个包含关键帧ID和位姿的字典列表。
        """
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{agent_rank}')
        poses_path = os.path.join(agent_output_dir, 'key_est_poses.npy')
        timestamps_path = os.path.join(agent_output_dir, 'key_timestamps.npy')

        if not os.path.exists(poses_path) or not os.path.exists(timestamps_path):
            print(f"[WARN] Mapper {self.rank}: Pose/timestamp files not found for agent {agent_rank}.")
            return []

        try:
            poses = np.load(poses_path)
            timestamps = np.load(timestamps_path)
        except Exception as e:
            print(f"[ERROR] Mapper {self.rank}: Failed to load pose/timestamp files for agent {agent_rank}: {e}")
            return []

        keyframes_in_bound = []
        for i, pose in enumerate(poses):
            position =pose [:3, 3]
            # 检查位置是否在边界框内
            if (position[0] >= bound[0, 0] and position[0] <= bound[0, 1] and
                position[1] >= bound[1, 0] and position[1] <= bound[1, 1] and
                position[2] >= bound[2, 0] and position[2] <= bound[2, 1]):
                keyframes_in_bound.append({'kf_id': int(timestamps[i]), 'pose': torch.from_numpy(pose)})
        # print(keyframes_in_bound)
        return keyframes_in_bound

    def find_multiple_matches(self, local_descriptors, foreign_descriptors, mutual_best=True):
        """
        在两组描述子之间找到最佳匹配对。
        在两组描述子之间找到多个高置信度的匹配对
        """
        if not local_descriptors or not foreign_descriptors:
            return None

        local_des_tensor = torch.cat([item['descriptor'] for item in local_descriptors], dim=0).to(self.device)
        foreign_des_tensor = torch.cat([item['descriptor'] for item in foreign_descriptors], dim=0).to(self.device)

        # 计算余弦相似度矩阵
        sim_matrix = F.cosine_similarity(local_des_tensor.unsqueeze(1), foreign_des_tensor.unsqueeze(0), dim=2)

        # 策略: 相互最佳匹配 (Mutual Best Match)
        # 从 local -> foreign 的最佳匹配
        best_foreign_for_local_val, best_foreign_for_local_idx = sim_matrix.max(dim=1)
        # 从 foreign -> local 的最佳匹配
        best_local_for_foreign_val, best_local_for_foreign_idx = sim_matrix.max(dim=0)

        matches = []
        for i in range(len(local_descriptors)):
            # 检查是否超过阈值
            if best_foreign_for_local_val[i] < self.loop_detector.sim_threshold:
                continue
            
            # 检查是否是相互最佳匹配
            match_idx_in_foreign = best_foreign_for_local_idx[i]
            if best_local_for_foreign_idx[match_idx_in_foreign] == i:
                matches.append({
                    'local_kf_id': local_descriptors[i]['kf_id'],
                    'foreign_kf_id': foreign_descriptors[match_idx_in_foreign]['kf_id'],
                    'similarity': best_foreign_for_local_val[i].item()
                })
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

    def get_pose_from_disk(self, agent_rank, kf_id):
        """
        从磁盘加载特定关键帧的位姿。
        """
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{agent_rank}')
        poses_path = os.path.join(agent_output_dir, 'key_est_poses.npy')
        timestamps_path = os.path.join(agent_output_dir, 'key_timestamps.npy')

        if not os.path.exists(poses_path) or not os.path.exists(timestamps_path):
            print(f"[WARN] Mapper {self.rank}: Pose/timestamp files not found for agent {agent_rank}.")
            return None

        try:
            poses = np.load(poses_path)
            timestamps = np.load(timestamps_path)
            
            match_idx = np.where(timestamps == kf_id)[0]
            if len(match_idx) > 0:
                return torch.from_numpy(poses[match_idx[0]]).to(self.device)
            else:
                print(f"[WARN] Mapper {self.rank}: KF ID {kf_id} not found in timestamps for agent {agent_rank}.")
                return None
        except Exception as e:
            print(f"[ERROR] Mapper {self.rank}: Failed to load pose for KF ID {kf_id} for agent {agent_rank}: {e}")
            return None

    def save_keyframe_data_atomic(self):
        """
        Atomically saves the current keyframe poses and timestamps to disk.
        This ensures that other agents can safely read these files for loop closure.
        """
        agent_output_dir = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
        os.makedirs(agent_output_dir, exist_ok=True)

        poses_path = os.path.join(agent_output_dir, 'key_est_poses.npy')
        timestamps_path = os.path.join(agent_output_dir, 'key_timestamps.npy')
        
        # 改名为更清晰的 temp 文件名
        temp_poses_path = os.path.join(agent_output_dir, 'key_est_poses_tmp.npy')
        temp_timestamps_path = os.path.join(agent_output_dir, 'key_timestamps_tmp.npy')

        with self.video.get_lock():
            num_keyframes = self.video.counter.value
            # keyframe_poses = self.video.get_all_pose(self.device)[:num_keyframes].cpu().numpy()
            keyframe_timestamps = self.video.timestamp[:num_keyframes].cpu().numpy()
            # 确定此映射迭代要使用的位姿
            if self.aligned_poses_c2w is not None:
                # 如果已发生回环闭合，则使用全局对齐后的位姿
                keyframe_poses = self.aligned_poses_c2w[:num_keyframes].cpu().numpy()
                print(f"[Mapper {self.rank}] Using previously aligned poses for mapping.")
            else:
                # 否则，从 video 缓冲区获取当前位姿
                keyframe_poses = self.video.get_all_pose(self.device)[:num_keyframes].cpu().numpy()

        np.save(temp_poses_path, keyframe_poses)
        os.replace(temp_poses_path, poses_path)

        np.save(temp_timestamps_path, keyframe_timestamps)
        os.replace(temp_timestamps_path, timestamps_path)

    def final_run(self):
        # with self.keyframe_dict_lock:
        #     current_keyframe_list = list(self.keyframe_dict)
        #     for i in range(len(current_keyframe_list)):
        #         kf_data = current_keyframe_list[i]
        #         original_c2w = kf_data['est_c2w'].to(self.device)
        #         aligned_c2w = self.relative_transform @ original_c2w
        #         kf_data['est_c2w'] = aligned_c2w.cpu()
        #         self.keyframe_dict[i] = kf_data
        # print(f"Agent {self.rank}: Applied transformation to its {num_keyframes} keyframes.")

        self.slam.save_checkpoint_before_fusion()
        # if self.slam.tracking_finished > 0 and not self.final_fusion_done:
        if not self.final_fusion_done:
            self.final_fusion_done = True # 确保此逻辑只执行一次

            if not self.use_bound_overlap or self.world_size <= 1:
                print(f"[Mapper {self.rank}] Final run: Skipping bound-based fusion.")
            else:
                print(f"[Mapper {self.rank}] Final run: Starting bound-based fusion process.")
                # self.fused_agents 集合包含了在 run() 阶段通过回环检测连接上的智能体rank。
                for other_rank in self.fused_agents:
                    print(f"[Mapper {self.rank}] Checking for overlap with Agent {other_rank}...")
                    print(self.all_agent_bounds)
                    # 计算重叠区域
                    current_agent_bound = self.all_agent_bounds[self.rank]
                    other_agent_bound = self.all_agent_bounds.get(other_rank)
                    if other_agent_bound is None:
                        print(f"[WARN] Mapper {self.rank}: Could not find bound for Agent {other_rank}. Skipping.")
                        continue
                    overlap_bound = self.compute_overlap_bound(current_agent_bound, other_agent_bound)
                    if overlap_bound is None:
                        print(f"[Mapper {self.rank}] No geometric overlap with Agent {other_rank}.")
                        continue
                    
                    print(f"[Mapper {self.rank}] Found geometric overlap with Agent {other_rank}.")

                    # 从重叠区域中获取两边的候选关键帧
                    local_kfs_in_overlap = self.get_keyframes_in_bound(self.rank, overlap_bound)
                    foreign_kfs_in_overlap = self.get_keyframes_in_bound(other_rank, overlap_bound)
                    if not local_kfs_in_overlap or not foreign_kfs_in_overlap:
                        print(f"[Mapper {self.rank}] Not enough keyframes in overlap region with Agent {other_rank}.")
                        continue

                    local_kf_ids = {kf['kf_id'] for kf in local_kfs_in_overlap}
                    foreign_kf_ids = {kf['kf_id'] for kf in foreign_kfs_in_overlap}

                    # 在重叠区域的关键帧之间寻找最佳的描述子匹配
                    with self.db_lock:
                        db_copy = list(self.slam.descriptor_db)
                    
                    local_descriptors = [item for item in db_copy if item['agent_id'] == self.rank and item['kf_id'] in local_kf_ids]
                    foreign_descriptors = [item for item in db_copy if item['agent_id'] == other_rank and item['kf_id'] in foreign_kf_ids]

                    # Find multiple matches instead of just the best one
                    matches = self.find_multiple_matches(local_descriptors, foreign_descriptors)
                    if len(matches) > self.config.get('distillation', {}).get('min_matches_for_fusion', 3):
                        print(f"[Mapper {self.rank}] Found {len(matches)} high-confidence matches with Agent {other_rank}.")

                        # 从匹配中找到外来智能体关键帧ID的范围扩展用于蒸馏的关键帧样本
                        foreign_kf_ids_from_matches = [m['foreign_kf_id'] for m in matches]
                        min_foreign_id = min(foreign_kf_ids_from_matches)
                        max_foreign_id = max(foreign_kf_ids_from_matches)
                        print(f"Agent {self.rank}: Expanding sample range from foreign KF ID {min_foreign_id} to {max_foreign_id}.")

                        # 从已知的在重叠区域内的外来关键帧中，筛选出ID在该范围内的所有帧
                        # foreign_kfs_in_overlap is a list of dicts {'kf_id': int, 'pose': torch.Tensor}
                        expanded_foreign_kfs_for_distill = [
                            kf for kf in foreign_kfs_in_overlap
                            if min_foreign_id <= kf['kf_id'] <= max_foreign_id
                        ]
                        num_expanded_kfs = len(expanded_foreign_kfs_for_distill)
                        print(f"Agent {self.rank}: Expanded from {len(matches)} matches to {num_expanded_kfs} keyframes for distillation.")
                        print(expanded_foreign_kfs_for_distill)
                        # Get pose pairs for alignment
                        pose_pairs = []
                        for match in matches:
                            local_pose = self.get_pose_from_disk(self.rank, match['local_kf_id'])
                            foreign_pose = self.get_pose_from_disk(other_rank, match['foreign_kf_id'])
                            if local_pose is not None and foreign_pose is not None:
                                pose_pairs.append((local_pose, foreign_pose))

                        if len(pose_pairs) < self.config.get('distillation', {}).get('min_matches_for_fusion', 3):
                            print(f"[WARN] Mapper {self.rank}: Not enough valid pose pairs ({len(pose_pairs)}) for alignment. Skipping.")
                            continue

                        print(f"[Mapper {self.rank}] Computed alignment transform to Agent {other_rank}.")

                        # Execute map distillation for all foreign keyframes in the overlap
                        # Load the foreign agent's model
                        other_agent_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{other_rank}')
                        other_agent_checkpoint_path = os.path.join(other_agent_path, 'latest_checkpoint.pt')
                        print(f"Agent {self.rank}: Loading other agent's model from {other_agent_checkpoint_path} for final fusion.")
                        other_agent_checkpoint = torch.load(other_agent_checkpoint_path, map_location=self.device)
                        foreign_bound = other_agent_checkpoint['bound'].to(self.device)
                        foreign_bounding_box = other_agent_checkpoint['bounding_box'].to(self.device)
                        self.model_shared.bound = foreign_bound
                        self.model_shared.bounding_box = foreign_bounding_box
                        self.model_shared.load_state_dict(other_agent_checkpoint['model'])
                        self.model_shared.all_planes = other_agent_checkpoint['all_planes']
                        self.model_shared.eval()

                        # 可视化蒸馏前的结果
                        if expanded_foreign_kfs_for_distill and matches:
                            # 使用最佳匹配的本地帧作为可视化的参考上下文
                            best_match = matches[0]
                            local_ref_kf_id = best_match['local_kf_id']
                            
                            # 一次性加载本地参考帧的数据
                            batch = self.dataset[local_ref_kf_id]
                            for k, v in batch.items():
                                batch[k] = v[None, ...] if isinstance(v, torch.Tensor) else torch.tensor([v])

                            # 遍历所有扩展的外部关键帧进行可视化
                            for kf_data in expanded_foreign_kfs_for_distill:
                                foreign_pose = kf_data['pose'].to(self.device)
                                foreign_kf_id = kf_data['kf_id']

                                self._visualize_model_output(
                                    c2w1=foreign_pose,
                                    foreign_frame_id=foreign_kf_id,
                                    title="Before"
                                )

                        # 多帧联合蒸馏
                        print(f"Agent {self.rank}: Starting joint distillation of {num_expanded_kfs} keyframe pairs with Agent {other_rank}.")

                        # 使用合并的数据进行蒸馏
                        for i in range(self.config['mapping']['distill_iters']):
                            # 在每次迭代中重新采样光线
                            all_rays_o = []
                            all_rays_d = []
                            all_target_rgb = []
                            all_target_depth = []

                            # 平均分配采样数量给每一对匹配
                            sample_per_match = max(self.config['mapping']['sample'] // num_expanded_kfs,
                                                self.config['mapping']['min_pixels_cur']) if num_expanded_kfs > 0 else self.config['mapping']['sample']

                            # 获取所有匹配的关键帧的位姿和数据
                            for kf_data in expanded_foreign_kfs_for_distill:
                                # 直接从预加载的数据中获取位姿，而不是再次从磁盘读取
                                foreign_pose = kf_data['pose'].to(self.device)

                                # 从 foreign_pose 采样光线
                                H, W = self.config['cam']['H'], self.config['cam']['W']
                                rays_d_cam = self.dataset.rays_d.reshape(-1, 3)
                                sample_indices = torch.randint(0, len(rays_d_cam), (sample_per_match,))
                                rays_d_cam_batch = rays_d_cam[sample_indices].to(self.device)

                                rays_o = foreign_pose[:3, 3].unsqueeze(0).repeat(sample_per_match, 1)
                                rays_d = torch.sum(rays_d_cam_batch[..., None, :] * foreign_pose[:3, :3], dim=-1)

                                all_rays_o.append(rays_o)
                                all_rays_d.append(rays_d)

                                # 从 teacher model 获取伪标签
                                with torch.no_grad():
                                    teacher_ret = self.model_shared.render_rays(rays_o, rays_d, target_d=None)
                                    all_target_rgb.append(teacher_ret['rgb'].detach())
                                    all_target_depth.append(teacher_ret['depth'].detach().unsqueeze(-1))

                            # 如果没有收集到有效数据，跳过蒸馏
                            if not all_rays_o:
                                print(f"[WARN] No valid data collected for joint distillation with Agent {other_rank} in iter {i+1}. Skipping.")
                                continue

                            # 合并所有数据
                            all_rays_o = torch.cat(all_rays_o, dim=0)
                            all_rays_d = torch.cat(all_rays_d, dim=0)
                            all_target_rgb = torch.cat(all_target_rgb, dim=0)
                            all_target_depth = torch.cat(all_target_depth, dim=0)

                            self.map_optimizer.zero_grad()
                            
                            student_ret = self.model.forward(all_rays_o, all_rays_d, all_target_rgb, all_target_depth)
                            distillation_loss = self.slam.get_loss_from_ret(student_ret, is_co_sdf=self.config['is_co_sdf'])

                            print(f"Agent {self.rank} Joint Distillation (with Agent {other_rank}) Iter {i+1}/{self.config['mapping']['distill_iters']} | Loss: {distillation_loss.item():.6f}")

                            distillation_loss.backward()
                            self.map_optimizer.step()
                            
                        print(f"Agent {self.rank}: Joint distillation with Agent {other_rank} complete.")

                        # 可视化蒸馏后的结果
                        if expanded_foreign_kfs_for_distill and matches:
                            # 同样，使用最佳匹配的本地帧作为可视化的参考上下文
                            best_match = matches[0]
                            local_ref_kf_id = best_match['local_kf_id']

                            # 一次性加载本地参考帧的数据
                            batch = self.dataset[local_ref_kf_id]
                            for k, v in batch.items():
                                batch[k] = v[None, ...] if isinstance(v, torch.Tensor) else torch.tensor([v])

                            # 遍历所有扩展的外部关键帧进行可视化
                            for kf_data in expanded_foreign_kfs_for_distill:
                                foreign_pose = kf_data['pose'].to(self.device)
                                foreign_kf_id = kf_data['kf_id']

                                self._visualize_model_output(
                                    c2w1=foreign_pose,
                                    foreign_frame_id=foreign_kf_id,
                                    title="After"
                                )

                        # 6.3. Save the final fused mesh
                        print(f"Agent {self.rank}: Saving final fused mesh (after joint distillation) with Agent {other_rank}.")
                        self.slam.save_mesh(f'agent_{self.rank}', f'final_fused_with_{other_rank}', voxel_size=self.config['mesh']['voxel_final'])

                    else:
                        print(f"[Mapper {self.rank}] Not enough matches found in overlap with Agent {other_rank}.")

        # 即使执行了融合，也可能需要处理最后一帧的建图
        print(f"[Mapper {self.rank}] Final run: Processing the very last keyframe.")

        print('final!')
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

        # self.global_BA(batch)
        # cur_c2w = self.c2w_matrices[self.N-1]
        # 确定此映射迭代要使用的位姿
        if self.aligned_poses_c2w is not None:
            # 如果已发生回环闭合，则使用全局对齐后的第 N 个位姿
            poses = self.aligned_poses_c2w[:self.N]
            print(f"[Mapper {self.rank}] Using aligned pose #{self.N} for mapping.")
        else:
            # 否则，从 video 缓冲区获取当前位姿
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

        # 为第一个关键帧计算并添加描述子到数据库
        if self.config['enable_loop_detect']:
            print(f"Agent {self.rank}: Adding descriptor for first keyframe {batch['frame_id']}...")
            # 注意：这里的batch没有额外的维度，所以直接用 batch['rgb']
            self.loop_detector.detect_and_add(
                current_kf_id=batch['frame_id'],
                current_agent_id=self.rank,
                frame_rgb=batch['rgb'][0]
            )
        # self.keyframe_dict.append(
        #     {'color': batch['rgb'][0].to(self.device), 'depth': batch['depth'][0].to(self.device),
        #      'est_c2w': cur_c2w.clone()})

        # c2w_array = np.array([tensor.cpu().numpy() for tensor in self.c2w_matrices])
        # Atomically save keyframe data for other agents to access for loop closure.
        self.save_keyframe_data_atomic()

        # self.update_poses()

        # mesh_save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.rank}')
        # mesh_directory = os.path.join(mesh_save_path, 'mesh')

        # # Create the directory if it doesn't exist
        # if not os.path.exists(mesh_directory):
        #     os.makedirs(mesh_directory)

        # mesh_out_file = os.path.join(mesh_directory, f'{idx:05d}_mesh.ply')

        # with self.keyframe_dict_lock:
        #     keyframe_dict_on_cpu = list(self.keyframe_dict)

        #     keyframe_dict_on_gpu = []
        #     for keyframe in keyframe_dict_on_cpu:
        #         keyframe_dict_on_gpu.append({
        #             'color': keyframe['color'].to(self.device),
        #             'depth': keyframe['depth'].to(self.device),
        #             'est_c2w': keyframe['est_c2w'].to(self.device)
        #         })

        # print("Getting mesh!")
        # self.mesher.get_mesh(mesh_out_file, keyframe_dict_on_gpu, self.device)
        
        # suffix = "occlusion"
        # cull_save_path = mesh_out_file.replace(".ply", "_cull_{}.ply".format(suffix))
        # cull_one_mesh(self.config, poses.cpu().numpy(), mesh_out_file, cull_save_path, save_unseen=False,
        #     remove_occlusion=True, scene_bounds=None,
        #     eps=0.03, th_obs=0, silent=True, platform='egl')
