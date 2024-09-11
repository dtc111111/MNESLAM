import argparse
import os
import random
import time

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

'''
reconstruction evaluation tools
modified from https://github.com/cvg/nice-slam/blob/master/src/tools/eval_recon.py
'''


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # for open3d 0.9.0
    # reg_p2p = o3d.registration.registration_icp(
    #     o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_mesh_metric(mesh_rec, mesh_gt, align=False):
    """
    3D reconstruction metric.

    """

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %

    return {'acc': accuracy_rec, 'comp': completion_rec, 'comp%': completion_ratio_rec}


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    print('accuracy: ', accuracy_rec)
    print('completion: ', completion_rec)
    print('completion ratio: ', completion_ratio_rec)

    return{
        'acc': accuracy_rec,
        'comp': completion_rec,
        'comp ratio': completion_ratio_rec
    }


def get_cam_position(gt_meshfile, sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    mesh_gt = trimesh.load(gt_meshfile)
    # Tbw: world_to_bound, bound is defined at the centre of cuboid
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= sz
    extents[1] *= sy
    extents[0] *= sx
    # Twb: bound_to_world
    transform = np.linalg.inv(to_origin)
    transform[0, 3] += dx
    transform[1, 3] += dy
    transform[2, 3] += dz
    return extents, transform


def calc_2d_metric(rec_meshfile, gt_meshfile, unseen_gt_pcd_file,
                   pose_file=None, gt_depth_render_file=None,
                   depth_render_file=None, suffix="virt_cams", align=True,
                   n_imgs=1000, not_counting_missing_depth=True,
                   sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    """
    2D reconstruction metric, depth L1 loss. modified from NICE-SLAM
    :param rec_meshfile: path to culled reconstructed mesh .ply
    :param gt_meshfile: path to culled GT mesh .ply
    :param unseen_gt_pcd_file: path to unseen pointcloud file .npy
    :param pose_file: path to sampled camera poses, saved as .npz (optional). Redo sampling if not provided
    :param gt_depth_render_file: path to rendered depth maps of GT mesh, saved as .npz (optional). Re-render if not provided
    :param depth_render_file: path to rendered depth maps of reconstructed mesh, saved as .npz (optional). Re-render if not provided
    :param suffix: suffix of reconstructed mesh
    :param align:
    :param n_imgs: number of views to sample
    :param not_counting_missing_depth: remove missing depth pixels in GT depth maps when computing depth L1
    :param sx: scale_x
    :param sy: scale_y
    :param sz: scale_z
    :param dx: offset_x
    :param dy: offset_y
    :param dz: offset_z
    :return:
    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    pc_unseen = np.load(unseen_gt_pcd_file)

    if pose_file and os.path.exists(pose_file):
        sampled_poses = np.load(pose_file)["poses"]
        assert len(sampled_poses) == n_imgs
        print("Found saved renering poses! Loading from disk!!!")
    else:
        sampled_poses = None
        print("Saved renering poses NOT FOUND! Will do the sampling")
    if gt_depth_render_file and os.path.exists(gt_depth_render_file):
        gt_depth_renderings = np.load(gt_depth_render_file)["depths"]
        assert len(gt_depth_renderings) == n_imgs
        print("Found saved renered gt depths! Loading from disk!!!")
    else:
        gt_depth_renderings = None
        print("Saved renered gt depths NOT FOUND! Will re-render!!!")
    if depth_render_file and os.path.exists(depth_render_file):
        depth_renderings = np.load(depth_render_file)["depths"]
        assert len(depth_renderings) == n_imgs
        print("Found saved renered reconstructed depth! Loading from disk!!!")
    else:
        depth_renderings = None
        print("Saved renered reconstructed depth NOT FOUND! Will re-render!!!")

    gt_dir = os.path.dirname(unseen_gt_pcd_file)
    log_dir = os.path.dirname(rec_meshfile)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile, sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    poses = []
    gt_depths = []
    depths = []
    for i in range(n_imgs):
        if sampled_poses is None:
            while True:
                # sample view, and check if unseen region is not inside the camera view
                # if inside, then needs to resample
                # camera-up (Y-direction) vector under world
                up = [0, 0, -1]
                # camera origin coord under world coordinate-frame, sampled within extents of the oriented bound
                origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform)
                origin = origin.reshape(-1)
                # sampled target coord under world [tx, ty, tz]
                tx = round(random.uniform(-10000, +10000), 2)
                ty = round(random.uniform(-10000, +10000), 2)
                tz = round(random.uniform(-10000, +10000), 2)
                target = [tx, ty, tz]
                # look_at vector (camera-Z), from origin to target
                target = np.array(target)-np.array(origin)
                c2w = viewmatrix(target, up, origin)
                tmp = np.eye(4)
                tmp[:3, :] = c2w
                c2w = tmp
                seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
                if (~seen):
                    break
            poses.append(c2w)
        else:
            c2w = sampled_poses[i]

        param = o3d.camera.PinholeCameraParameters()
        # extrinsic is w2c
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        if gt_depth_renderings is None:
            vis.add_geometry(gt_mesh, reset_bounding_box=True,)
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            gt_depth = vis.capture_depth_float_buffer(True)
            gt_depth = np.asarray(gt_depth)
            vis.remove_geometry(gt_mesh, reset_bounding_box=True,)
            gt_depths.append(gt_depth)
        else:
            gt_depth = gt_depth_renderings[i]
        
        if depth_renderings is None:
            vis.add_geometry(rec_mesh, reset_bounding_box=True,)
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            ours_depth = vis.capture_depth_float_buffer(True)
            ours_depth = np.asarray(ours_depth)
            vis.remove_geometry(rec_mesh, reset_bounding_box=True,)
            depths.append(ours_depth)
        else:
            ours_depth = depth_renderings[i]

        if not_counting_missing_depth:
            valid_mask = (gt_depth > 0.) & (gt_depth < 19.)
            if np.count_nonzero(valid_mask) <= 100:
                continue
            # print(i, np.count_nonzero(valid_mask))
            errors += [np.abs(gt_depth[valid_mask] - ours_depth[valid_mask]).mean()]
        else:
            errors += [np.abs(gt_depth-ours_depth).mean()]

    if pose_file is None:
        np.savez(os.path.join(gt_dir, "sampled_poses_{}.npz".format(n_imgs)), poses=poses)
    elif not os.path.exists(pose_file):
        np.savez(pose_file, poses=poses)

    if gt_depth_render_file is None:
        np.savez(os.path.join(gt_dir, "gt_depths_{}.npz".format(n_imgs)), depths=gt_depths)
    elif not os.path.exists(gt_depth_render_file):
        np.savez(gt_depth_render_file, depths=gt_depths)

    if depth_render_file is None:
        np.savez(os.path.join(log_dir, "depths_{}_{}.npz".format(suffix, n_imgs)), depths=depths)
    elif not os.path.exists(depth_render_file):
        np.savez(depth_render_file, depths=depths)

    errors = np.array(errors)
    # from m to cm
    print('Depth L1: ', errors.mean() * 100)
    return {"Depth L1": errors.mean() * 100}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to evaluate the reconstruction."
    )
    parser.add_argument("--rec_mesh", type=str,
                        help="reconstructed mesh file path")
    parser.add_argument("--gt_mesh", type=str,
                        help="ground truth mesh file path")
    parser.add_argument("--dataset_type", type=str, default="Replica",
                        help="dataset type: [Replica, RGBD]")
    parser.add_argument("-2d", "--metric_2d",
                        action="store_true", help="enable 2D metric")
    parser.add_argument("-3d", "--metric_3d",
                        action="store_true", help="enable 3D metric")
    args = parser.parse_args()

    if args.metric_3d:
        calc_3d_metric(args.rec_mesh, args.gt_mesh)

    if args.metric_2d:
        assert args.dataset_type in ["Replica", "RGBD"], "Unknown dataset type..."
        eval_data_dir = os.path.dirname(args.gt_mesh)
        unseen_pc_file = os.path.join(eval_data_dir, "gt_pc_unseen.npy")
        pose_file = os.path.join(eval_data_dir, "sampled_poses_1000.npz")
        if args.dataset_type == "Replica":  # follow NICE-SLAM
            sx, sy, sz, dx, dy, dz = 0.3, 0.7, 0.7, 0.0, 0.0, 0.4
        elif os.path.basename(eval_data_dir) == "complete_kitchen":  # complete_kitchen has special shape
            sx, sy, sz, dx, dy, dz = 0.3, 0.5, 0.5, 1.2, 0.0, 1.8
        else:
            sx, sy, sz, dx, dy, dz = 0.3, 0.6, 0.6, 0.0, 0.0, 0.0
        calc_2d_metric(args.rec_mesh, args.gt_mesh, unseen_pc_file, pose_file=pose_file, n_imgs=1000,
                       not_counting_missing_depth=True, sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz)
