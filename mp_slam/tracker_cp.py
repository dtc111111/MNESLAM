import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


import os
import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, Style
from collections import OrderedDict
from tqdm import tqdm
# from lietorch import SE3
# from time import gmtime, strftime, time, sleep
# import torch.multiprocessing as mp
#
# from .droid_net import DroidNet
# from .frontend import Frontend
# from .backend import Backend
# from .depth_video import DepthVideo
# from .motion_filter import MotionFilter
# from .multiview_filter import MultiviewFilter
# from .visualization import droid_visualization



class Tracker(nn.Module):

    def __init__(self, cfg, args, slam):
        super(Tracker, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = args.device
        self.net = slam.net
        self.video = slam.video
        self.verbose = slam.verbose

        # filter incoming frames so that there is enough motion
        self.frontend_window = cfg['tracking']['frontend']['window']
        filter_thresh = cfg['tracking']['motion_filter']['thresh']
        # self.motion_filter = MotionFilter(self.net, self.video, thresh=filter_thresh, device=self.device)
        #
        # # frontend process
        # self.frontend = Frontend(self.net, self.video, self.args, self.cfg)
    
    def tracking_render(self, batch, frame_id):

        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''



        with torch.no_grad():
            ### check there is enough motion
            self.motion_filter.track(frame_id, batch['rgb'], batch['depth'], batch['intrinsic'], gt_pose=batch['c2w'])
            # local bundle adjustment
            self.frontend()

    def run(self):
        # for idx, batch in tqdm(enumerate(self.data_loader)):
        #     if idx == 0:
        #         continue
        #     while self.mapping_idx[0] < idx-self.config['mapping']['map_every']-\
        #         self.config['mapping']['map_every']//2:
        #         time.sleep(0.1)
        #
        #     self.update_params()
        #     self.tracking_render(batch, idx)
        #     self.tracking_idx[0] = idx

        for idx, batch in tqdm(enumerate(self.data_loader)):  # slam.py def tracking
            if idx == 0:
                continue
            while self.mapping_idx[0] < idx - self.config['mapping']['map_every'] - \
                    self.config['mapping']['map_every'] // 2:
                time.sleep(0.1)

            self.tracker_render(batch, idx)
            print('tracking finished')


        


            

