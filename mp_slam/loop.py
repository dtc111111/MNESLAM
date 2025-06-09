




if current_map_id == self.config['loop_kf2_id']:
    ############################### step1 ####### 蒸馏 #########################################
    # load ckpt
    save_path = os.path.join(self.config['data']['output'], self.config['data']['exp_name_1'])
    poses_all = np.load(f'{save_path}/key_est_poses.npy')  # tuple1100,4,4

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
            indices].to(self.device)

        # N pose Bs rays
        N = 20
        indices = torch.randint(0, len(poses_all), (N,))
        selected_poses = poses_all[indices]

        selected_poses = torch.stack(
            [torch.tensor(pose, dtype=torch.float32) for pose in selected_poses]).to(
            self.device)

        # 相机坐标系下的射线方向 N,Bs,1,3 * N,1,3,3 = N,Bs,3
        rays_d = torch.sum(rays_d_cam[None, :, None, :] * selected_poses[:, None, :3, :3], -1)

        rays_o = selected_poses[..., None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        # torch.Size([92160, 3])

        agent1_rendered_ret = self.model_shared.render_rays(rays_o, rays_d, target_d=None, loop=True)
        agent1_rendered_rgb, agent1_rendered_depth = agent1_rendered_ret['rgb'], agent1_rendered_ret['depth']
        # [bs,3], [bs,1]

        agent1_rendered_depth = agent1_rendered_depth.unsqueeze(1)

        ret = self.model.forward(rays_o, rays_d, agent1_rendered_rgb, agent1_rendered_depth)
        loss = self.slam.get_loss_from_ret(ret, is_co_sdf=self.config['is_co_sdf'])

        loss.backward()
        self.map_optimizer.step()

        if i % 5 == 0:
            self.slam.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])
            self.slam.save_imgs(i, batch['depth'][0], batch['rgb'][0], cur_c2w)

    idx = int(self.mapping_idx[0].item())
    print('Save loop mesh!', idx)

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

self.slam.save_imgs(99999, batch['depth'][0], batch['rgb'][0], loop_a_pose)

# 从 b 坐标系到 a 坐标系的变换矩阵
loop_b2a_pose = loop_a_pose @ cur_c2w.inverse()

print('loop_pose', loop_b2a_pose)