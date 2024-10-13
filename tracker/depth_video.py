import torch
import lietorch
import droid_backends
from copy import deepcopy

from torch.multiprocessing import Value

from .droid_net import cvx_upsample
from .geom import projective_ops as pops

from model.keyframe import KeyFrameDatabase
class DepthVideo:
    def __init__(self, cfg, slam):
        self.cfg = cfg

        self.keyframe = slam.keyframeDatabase

        # current keyframe count
        self.counter = Value('i', 0)

        self.ready = Value('i', 0)
        self.mapping = Value('i', 0)
        self.ba_lock = {
            'dense': Value('i', 0),
            'loop': Value('i', 0),
        }
        self.global_ba_lock = Value('i', 0)
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd

        self.map_counter = Value('i', 0)
########################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        c = 1
        self.scale_factor = 8
        s = self.scale_factor
        buffer = cfg['tracking']['buffer']

        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        #
        self.rm_timestamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.rm_current_length = Value('i', 0)
        #
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.float)
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.poses_gt = torch.zeros(buffer, 4, 4, device=device, dtype=torch.float).share_memory_()  # c2w matrix
        self.disps = torch.ones(buffer, ht//s, wd//s, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//s, wd//s, device=device, dtype=torch.float).share_memory_()
        self.depths_gt = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//s, wd//s, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//s, wd//s, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//s, wd//s, dtype=torch.half, device=device).share_memory_()

        ### initialize poses to identity transformation
        self.poses[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.poses_gt[:] = torch.eye(4, dtype=torch.float, device=device)

        ### consistent construction
        self.poses_filtered = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()  # w2c quaterion
        self.disps_filtered = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.mask_filtered = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.poses_filtered[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.filtered_id = torch.tensor([-1, ], dtype=torch.int, device=device).share_memory_()
        self.update_priority = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.bound = torch.zeros(1, 3, 2, device=device, dtype=torch.float).share_memory_()

        ### pose compensation from vitural to real
        self.pose_compensate = torch.zeros(1, 7, dtype=torch.float, device=device).share_memory_()
        self.pose_compensate[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        # self.pose_compensate[:] = torch.tensor([0.6763, -1.0931, 3.2909, 0.5002, 0.6974, -0.4171, 0.2992],
        #                                        dtype=torch.float, device=device)

    def get_lock(self):
        return self.counter.get_lock()

    def get_ba_lock(self, ba_type):
        return self.ba_lock[ba_type].get_lock()

    def get_mapping_lock(self):
        return self.mapping.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1]


        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            self.depths_gt[index] = item[4]
            depth = item[4][..., 3::8, 3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)
            self.disps[index] = self.disps_sens[index].clone()

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.poses_gt[index] = item[9].to(self.poses_gt.device)

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index > 0:
                index = self.counter.value + index
            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
            )

        return item

    def append(self, *item):
         with self.get_lock():
             self.__item_setter(self.counter.value, item)

    def get_bound(self):
        with self.mapping.get_lock():
            bound = self.bound[0]

        return bound

    ###  dense mapping operations ###
    def get_mapping_item(self, index, device='cuda:0', decay=0.1):
        with self.mapping.get_lock():
            image = self.images[index].clone().permute(1, 2, 0).contiguous().to(device)  # [h, w, 3]
            mask = self.mask_filtered[index].clone().to(device)
            est_disp = self.disps_filtered[index].clone().to(device)  # [h, w]
            # gt_depth = self.depths_gt[index].clone().to(device)  # [h, w]
            est_depth = 1.0 / (est_disp + 1e-7)

            # origin alignment
            w2c = lietorch.SE3(self.poses_filtered[index].clone()).to(device) # Tw(droid)_to_c
            c2w = lietorch.SE3(self.pose_compensate[0].clone()).to(w2c.device) * w2c.inv()
            c2w = c2w.matrix()  # [4, 4]

            gt_c2w = self.poses_gt[index].clone().to(device)  # [4, 4]

            depth = est_depth

            # if updated by mapping, the priority is decreased to lowest level, i.e., 0
            self.update_priority[index] *= decay

            return image, depth, c2w, gt_c2w, mask


    def get_pose(self, N, device='cuda:0'):

        # 初始化矩阵来存储转换后的位姿
        c2w_matrices = []
        print(self.poses)
        # 遍历所有位姿索引
        for index in range(N):
            # 从世界坐标到相机坐标的转换
            w2c = lietorch.SE3(self.poses[index].clone()).to(device)
            # 从相机坐标到世界坐标的转换，包括补偿位姿
            c2w = lietorch.SE3(self.pose_compensate[0].clone()).to(w2c.device) * w2c.inv()
            c2w = c2w.matrix()  # 转换为4x4矩阵

            trans = self.poses_gt[0].clone()
            trans[:3, 1] *= -1
            trans[:3, 2] *= -1
            # print('####################trans', trans)

            # trans = torch.tensor([[-3.2057e-01, 4.4806e-01, -8.3455e-01, 3.4530e+00],
            #                       [9.4722e-01, 1.5164e-01, -2.8244e-01, 4.5461e-01],
            #                       [1.0790e-16, -8.8105e-01, -4.7302e-01, 5.9363e-01],
            #                       [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]], device=device)

            c2w = trans @ c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1

            # 26 gt -3.856797169672012693e-01 3.021256147992357199e-01 -8.717633100803701129e-01 3.566249804253109446e+00 9.226327307873375405e-01 1.262948058485572900e-01 -3.644152385611442724e-01 6.892607707624622826e-01 1.157125811473194137e-16 -9.448649294464576132e-01 -3.274603259971212710e-01 4.649919160893953851e-01 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
            # print('c2w_trans', c2w)
            c2w_matrices.append(c2w)

        c2w_matrices_tensor = torch.stack(c2w_matrices)

        return c2w_matrices_tensor

    def set_item_from_mapping(self, index, pose=None, depth=None):
        with self.get_lock():
            pass

    ### geometric operations ###

    @staticmethod
    def format_indices(ii, jj, device='cuda'):
        """ to device, long, {-1}"""
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(dim=-1), mask) # [b, h, w, 1]
        self.disps_up[ix] = disps_up.squeeze()  # [b, h, w]

    def normalize(self):
        """ normalize depth and poses """
        with self.get_lock():
            cur_ix = self.counter.value
            s = self.disps[:cur_ix].mean()
            self.disps[:cur_ix] /= s
            self.poses[:cur_ix, :3] *= s  # [tx, ty, tz, qx, qy, qz, qw]
            self.dirty[:cur_ix] = True

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indices(ii, jj, self.device)
        Gs = lietorch.SE3(self.poses[None, ...])

        coords, valid_mask = pops.projective_transform(
            poses=Gs, depths=self.disps[None, ...], intrinsics=self.intrinsics[None, ...],
            ii=ii, jj=jj, jacobian=False, return_depth=False,
        )

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric, where distance = sqrt((u(ii) - u(jj->ii))^2 + (v(ii) - v(jj->ii))^2) """
        return_matrix = False
        N = self.counter.value
        if ii is None:
            return_matrix = True
            ii, jj = torch.meshgrid(
                torch.arange(N),
                torch.arange(N),
                indexing='ij'
            )

        ii, jj = DepthVideo.format_indices(ii, jj)

        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        if bidirectional:
            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta
            )

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[intrinsic_common_id], jj, ii, beta
            )

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[intrinsic_common_id], ii, jj, beta
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, ba_type=None):
        """ dense bundle adjustment (DBA) """
        intrinsic_common_id = 0  # we assume the intrinsic within one scene is the same
        lock = self.get_lock() if ba_type is None else self.get_ba_lock(ba_type)
        with lock:
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[intrinsic_common_id], self.disps_sens,
                              target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)

