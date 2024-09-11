import os
import numpy as np
import trimesh


def vis_mesh(scene_dir):
    import open3d as o3d
    from datasets.vis_cameras import draw_cuboid

    mesh_our = o3d.io.read_triangle_mesh(os.path.join(scene_dir, "scene/integrated_rot.ply"))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    xmin, ymin, zmin = -3.2, -3.1, -1.7
    xmax, ymax, zmax = 3.2, 2.6, 1.2
    bound = np.array([[xmin, xmax],
                      [ymin, ymax],
                      [zmin, zmax]])
    np.savetxt(os.path.join(scene_dir, "bound.txt"), bound)
    cube = draw_cuboid(bound)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True
    vis.add_geometry(mesh_our)
    # vis.add_geometry(mesh_nice)
    vis.add_geometry(frame)
    # vis.add_geometry(rot_frame)
    vis.add_geometry(cube)
    vis.run()


def align(mesh_dir):
    # up = np.array([0.986774, 0.155136, -0.047004])
    # right = np.array([0.141468, -0.985367, -0.095075])
    y, z = fit_plane(mesh_dir)
    # print(np.dot(y, z))
    x = np.cross(y, z)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=1)
    return rot


def fit_plane(mesh_dir):
    import pyransac3d as pyrsc
    floor_path = os.path.join(mesh_dir, "floor.ply")
    floor = trimesh.load(floor_path)
    floor_pts = floor.vertices
    floor_plane = pyrsc.Plane()
    floor_eq, _ = floor_plane.fit(floor_pts, 0.01)
    z = np.array(floor_eq[:3])
    # make sure z is pointing upward
    if z[1] > 0:
        z = -z

    wall_path = os.path.join(mesh_dir, "wall.ply")
    wall = trimesh.load(wall_path)
    wall_pts = wall.vertices
    wall_plane = pyrsc.Plane()
    wall_eq, _ = wall_plane.fit(wall_pts, 0.01)
    y = np.array(wall_eq[:3])

    return y, z


def align_mesh(scene_dir):
    mesh_dir = os.path.join(scene_dir, "scene")
    rot = align(mesh_dir)
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = np.linalg.inv(rot)
    mesh_path = os.path.join(mesh_dir, "integrated.ply")
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh.apply_transform(trans_mat)
    mesh_path = mesh_path.replace(".ply", "_rot.ply")
    mesh.export(mesh_path)
    np.savetxt(os.path.join(scene_dir, "align_mat.txt"), trans_mat)
    # compute bound
    xmin, xmax = np.min(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 0])
    ymin, ymax = np.min(mesh.vertices[:, 1]), np.max(mesh.vertices[:, 1])
    zmin, zmax = np.min(mesh.vertices[:, 2]), np.max(mesh.vertices[:, 2])
    bound = np.array([[xmin, xmax],
                      [ymin, ymax],
                      [zmin, zmax]])
    np.savetxt(os.path.join(scene_dir, "bound.txt"), bound)


if __name__ == "__main__":
    data_root = "/media/jingwen/Data1/nice-slam_data/my_room"
    scene = "11"
    scene_dir = os.path.join(data_root, scene)
    vis_mesh(scene_dir)
