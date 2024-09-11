import open3d as o3d

def voxel_down_sample(input_file):
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_file)

    # 执行体素网格下采样
    # downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    # 使用随机下采样
    downsampled = pcd.random_down_sample(sampling_ratio=0.7)  # 保留50%的点

    o3d.io.write_point_cloud('/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/ntu_S1_B1_indoor_delpp_ds07.pcd', pcd)

    return downsampled



def compute_normals_and_display(input_file, save_path):
    pcd = o3d.io.read_point_cloud(input_file)
    # 计算法线
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))

    o3d.io.write_point_cloud(save_path, pcd)

# 使用示例
# 替换下面的路径为你的 PCD 文件路径

# 替换以下路径为你的 PCD 文件路径
# downsampled_pcd = voxel_down_sample('/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/ntu_S1_B1_indoor_delpp.pcd')

compute_normals_and_display('/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/ntu_S1_B1_indoor_delpp.pcd', '/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/ntu_S1_B1_indoor_delpp_norm0250.pcd')
