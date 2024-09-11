import os
import argparse
import imageio
import numpy as np
import open3d as o3d
from tqdm import tqdm

from config import load_config
from datasets.dataset import get_dataset


def refresh(vis):
    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    w2c = np.array(cam.extrinsic)
    c2w = np.linalg.inv(w2c)
    print(c2w)
    return True


def save_c2w(vis):
    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    w2c = np.array(cam.extrinsic)
    c2w = np.linalg.inv(w2c)
    idx = len([f for f in os.listdir(vis_param["save_dir"]) if f.endswith("txt")])
    np.savetxt(os.path.join(vis_param["save_dir"], "{}.txt".format(idx)), c2w)
    print(c2w)
    image = vis.capture_screen_float_buffer(True)
    image = (np.asarray(image) * 255.).astype(np.uint8)
    imageio.imwrite(os.path.join(vis_param["save_dir"], "{}.png".format(idx)), image)
    return True


def run_tsdf_fusion(cfg, save_path):
    dataset = get_dataset(cfg)
    H, W = dataset.H, dataset.W
    fx, fy, cx, cy = dataset.fx, dataset.fy, dataset.cx, dataset.cy
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    voxel_length = 0.03
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.12,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i, frame in enumerate(tqdm(dataset)):
        rgb, depth, c2w = frame["rgb"].cpu().numpy(), frame["depth"].cpu().numpy(), frame["c2w"].cpu().numpy()
        rgb = rgb * 255
        rgb = rgb.astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        depth = depth.astype(np.float32)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0,
                                                                  depth_trunc=8.0,
                                                                  convert_rgb_to_intensity=False)
        # to OpenCV
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        # requires w2c
        w2c = np.linalg.inv(c2w)
        volume.integrate(rgbd, K, w2c)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print("Saving mesh...")
    o3d.io.write_triangle_mesh(save_path, mesh)


vis_param = {"save_dir": None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.", default="./configs/Replica/office0.yaml")
    parser.add_argument("--data_dir", type=str,
                        help="Path to dataset sequence. This has higher priority.", default="./data/Replica/office0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg["data"]["datadir"] = args.data_dir
    datadir = cfg["data"]["datadir"]
    scene = os.path.dirname(datadir)

    # create TSDF-Fusion
    tsdf_fusion_file = os.path.join(datadir, "tsdf_fusion.ply")
    if not os.path.exists(tsdf_fusion_file):
        print("TSDF-Fusion mesh not created, creating now...")
        run_tsdf_fusion(cfg, tsdf_fusion_file)

    # here, mesh to see if the selected novel views cover all the regions
    # obtain mesh and visualize
    mesh = o3d.io.read_triangle_mesh(tsdf_fusion_file)  # replace tsdf_fusion_file with the mesh with selected views
    mesh.compute_vertex_normals()
    vis_param["save_dir"] = os.path.join(datadir, "virtual_cameras".format(scene))
    if not os.path.exists(vis_param["save_dir"]):
        os.makedirs(vis_param["save_dir"])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1200, height=680)
    vis.get_render_option().mesh_show_back_face = True
    # set visualizer intrinsics
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    w, h = 1200, 680
    fx = 600.0
    fy = 600.0
    cx = 599.5
    cy = 339.5
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    ctr.convert_from_pinhole_camera_parameters(init_param)
    vis.register_key_callback(key=ord("."), callback_func=save_c2w)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(axis)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
