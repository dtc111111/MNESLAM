import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from .utils import get_camera_rays, alphanum_key, as_intrinsics_matrix


def get_dataset(config):
    '''
    Get the dataset class from the config file.
    '''
    if config['dataset'] == 'replica':
        dataset = ReplicaDataset

    elif config['dataset'] == 'scannet':
        dataset = ScannetDataset

    elif config['dataset'] == 'indoor':
        dataset = Indoor

    return dataset(config,
                   config['data']['datadir'],
                   trainskip=config['data']['trainskip'],
                   downsample_factor=config['data']['downsample'],
                   sc_factor=config['data']['sc_factor'])


class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H'] // cfg['data']['downsample'], \
                         cfg['cam']['W'] // cfg['data']['downsample']

        self.fx, self.fy = cfg['cam']['fx'] // cfg['data']['downsample'], \
                           cfg['cam']['fy'] // cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx'] // cfg['data']['downsample'], \
                           cfg['cam']['cy'] // cfg['data']['downsample']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size * 2) * (self.W - self.crop_size * 2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

class ReplicaDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0):
        super(ReplicaDataset, self).__init__(cfg)

        self.t0 = cfg['start_index']
        self.t1 = cfg['end_index']

        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/results/frame*.jpg'))[self.t0:self.t1]
        self.depth_paths = sorted(
            glob.glob(f'{self.basedir}/results/depth*.png'))[self.t0:self.t1]
        self.load_poses(os.path.join(self.basedir, 'traj.txt'))

        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,

        }

        return ret

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()[self.t0:self.t1]
        for i in range(len(self.img_files)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class ScannetDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0):
        super(ScannetDataset, self).__init__(cfg)
        self.t0 = cfg['start_index']
        self.t1 = cfg['end_index']
        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))[self.t0:self.t1]
        self.depth_paths = sorted(
            glob.glob(os.path.join(
                self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))[self.t0:self.t1]
        self.load_poses(os.path.join(self.basedir, 'pose'))

        # self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F, 5)

        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge'] * 2
            self.W -= self.config['cam']['crop_edge'] * 2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d
        }

        return ret

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
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

class Indoor(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0):
        super(Indoor, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.t0 = cfg['start_index']
        self.t1 = cfg['end_index']
        self.color_paths = sorted(
            glob.glob(os.path.join(self.basedir, 'color', '*.jpg')))[self.t0:self.t1]
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.basedir, 'depth_holefilling', '*.png')))[self.t0:self.t1]
      
        self.rays_d = None
        self.frame_ids = range(0, len(self.color_paths))
        self.num_frames = len(self.frame_ids)
        self.load_poses(f'{self.basedir}/traj.txt')

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge'] * 2
            self.W -= self.config['cam']['crop_edge'] * 2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):

        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('int16') / 65535. * 100.
        # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535. * 100.

        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        # depth_data[depth_data > 6] = 0

        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d
        }

        return ret

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()[self.t0:self.t1]

        trans = np.array([[0.970296, -0.241922, 0.000000, -0.789423],
                          [0.241922, 0.970296, 0.000000, -6.085402],
                          [0.000000, 0.000000, 1.000000, 0.000000],
                          [0.000000, 0.000000, 0.000000, 1.000000]])

        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w_transformed = np.dot(trans, c2w)
            c2w_tensor = torch.from_numpy(c2w_transformed).float()
            c2w_tensor[:3, 1] *= -1
            c2w_tensor[:3, 2] *= -1
            self.poses.append(c2w_tensor)
