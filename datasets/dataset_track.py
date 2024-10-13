import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset_track(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']

        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.n_img = -1
        self.depth_paths = None
        self.color_paths = None
        self.poses = None
        self.image_timestamps = None

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None

        self.input_folder = cfg['data']['datadir']

    def __len__(self):
        return self.n_img

    def depthloader(self, index):
        if self.depth_paths is None:
            return None
        depth_path = self.depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale

        return depth_data

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx, self.cx, self.fy, self.cy
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        outsize = (H_out_with_edge, W_out_with_edge)

        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]
        if self.name == 'indoor':

            depth_path = self.depth_paths[index]
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.int16) / 65535. * 100.

        else:
            depth_data = self.depthloader(index)

        if depth_data is not None:
            depth_data = torch.from_numpy(depth_data).float()
            depth_data = F.interpolate(
                depth_data[None, None], outsize, mode='nearest')[0, 0]

        intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
            depth_data = depth_data[:, edge:-edge]
            intrinsic[2] -= edge

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
            depth_data = depth_data[edge:-edge, :]
            intrinsic[3] -= edge

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, intrinsic, pose


class Replica(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Replica, self).__init__(cfg, device)
        stride = cfg['stride']
        self.t0 = cfg['start_index']
        self.t1 = cfg['end_index']
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))[self.t0:self.t1]
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))[self.t0:self.t1]
        self.n_img = len(self.color_paths)

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[::stride]
        self.depth_paths = self.depth_paths[::stride]
        self.poses = self.poses[::stride]
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()[self.t0:self.t1]
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            self.poses.append(c2w)


class Indoor(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(Indoor, self).__init__(cfg,device)
        self.t0 = cfg['start_index']
        self.t1 = cfg['end_index']
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))[self.t0:self.t1]
        # self.depth_paths = sorted(
        #     glob.glob(f'{self.input_folder}/results/depth*.png'))

        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth_filtered', '*.png')))[self.t0:self.t1]
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')


    # def load_poses(self, path):
    #     self.poses = []
    #     with open(path, "r") as f:
    #         lines = f.readlines()[self.t0:self.t1]
    #     for i in range(self.n_img):
    #         line = lines[i]
    #         c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
    #         c2w[:3, 1] *= -1
    #         c2w[:3, 2] *= -1
    #         self.poses.append(c2w)

    def load_poses(self, path):
        self.poses = []

        with open(path, "r") as f:
            lines = f.readlines()[self.t0:self.t1]

        # trans = np.array([[-0.231550753117, - 0.972703993320, 0.015204750933, 3.524853229523],
        #                   [0.044645648450, - 0.026238689199, - 0.998658597469, - 0.090221792459],
        #                   [0.971798300743, - 0.230560690165, 0.049501836300, 3.537426233292],
        #                   [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])

        trans = np.array([[0.970296, -0.241922, 0.000000, -0.789423],
                          [0.241922, 0.970296, 0.000000, -6.085402],
                          [0.000000, 0.000000, 1.000000, 0.000000],
                          [0.000000, 0.000000, 0.000000, 1.000000]])

        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w_transformed = np.dot(trans, c2w)
            c2w_transformed[:3, 1] *= -1
            c2w_transformed[:3, 2] *= -1
            self.poses.append(c2w_transformed)

class ScanNet(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, device)

        self.t0 = cfg['start_index']
        self.t1 = cfg['end_index']

        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))[self.t0:self.t1]
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))[self.t0:self.t1]
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))[self.t0:self.t1]
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            self.poses.append(c2w)


dataset_dict = {
    "replica": Replica,
    'scannet': ScanNet,
    'indoor': Indoor,
}
