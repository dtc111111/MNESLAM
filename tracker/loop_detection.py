import pandas as pd
import torch

from geom import projective_ops as pops
from modules.corr import CorrBlock

from tracker.droid_net import DroidNet
from collections import OrderedDict
import argparse
import config

class Loop_detector:
    def __init__(self, cfg, net, device='cuda:0'):
        self.net = net
        self.device = device
        self.cnet = net.cnet  # Context network part
        self.fnet = net.fnet  # Feature network part
        self.update = net.update  # Update module part

        # Normalization parameters
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]


        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
        self.fx, self.fy, self.cx, self.cy = cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def run(self, image1, image2):
        # Normalize and prepare images
        # img1 = image1.unsqueeze(dim=0).to(self.device).sub_(self.MEAN).div_(self.STDV)
        img1 = image1.unsqueeze(dim=0).to(self.device).clone()
        img1 = img1.sub(self.MEAN).div(self.STDV)

        img2 = image2.unsqueeze(dim=0).to(self.device).sub_(self.MEAN).div_(self.STDV)

        # Feature extraction for both images
        fmap1 = self.__feature_encoder(img1)  # [1, 128, h//8, w//8]
        fmap2 = self.__feature_encoder(img2)  # [1, 128, h//8, w//8]

        # Prepare correlation block and coords grid
        ht, wd = fmap1.shape[2], fmap1.shape[3]
        coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]  # [1, 1, h//8, w//8, 2]
        corr = CorrBlock(fmap1[None, [0]], fmap2[None, [0]])(coords0)  # [1, 1, 4*49, h//8, w//8]

        # Compute optical flow using update module
        _, delta, weight = self.update(fmap1[None], fmap2[None], corr)  # [1, 1, h//8, w//8, 2]

        return delta

    def __feature_encoder(self, image):
        """Assuming a pre-trained network that outputs 128-channel feature map reduced by 8x."""
        return self.fnet(image).squeeze(dim=0)

    def load_image(self, image_path):
        """Loads an image and converts it to a tensor."""

        color_data = cv2.imread(image_path)

        H_out_with_edge = self.H_out + self.H_edge * 2
        W_out_with_edge = self.W_out + self.W_edge * 2
        color_data = cv2.resize(color_data, (W_out_with_edge, H_out_with_edge))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)  # BGR到RGB
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1) / 255.0  # [C, H, W] 归一化到[0, 1]
        color_data = color_data.unsqueeze(dim=0)
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]


        return color_data.to(self.device)

    def find_minimum_flow(self, dataset1, dataset2, save_csv_path):
        # Initialize DataFrame
        results = pd.DataFrame(columns=['i', 'j', 'flow_magnitude'])
        min_flow = float('inf')
        min_indices = None

        for i, img1_path in enumerate(dataset1):
            image1 = self.load_image(img1_path)
            print(i)
            for j, img2_path in enumerate(dataset2):

                image2 = self.load_image(img2_path)
                delta = self.run(image1, image2)
                flow_magnitude = delta.norm(dim=-1).mean().item()
                results = results.append({'i': i, 'j': j, 'flow_magnitude': flow_magnitude}, ignore_index=True)

                if flow_magnitude < min_flow:
                    min_flow = flow_magnitude
                    min_indices = (i, j)


        print(f"Minimum flow magnitude: {min_flow} between images at indices {min_indices[0]} and {min_indices[1]}")
        results.to_csv(save_csv_path, index=False)
        return min_indices


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_pretrained(net, pretrained):
    print(f'INFO: load pretrained checkpiont from {pretrained}!')

    state_dict = OrderedDict([
        (k.replace('module.', ''), v) for (k, v) in torch.load(pretrained).items()
    ])

    state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
    state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
    state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
    state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]

    net.load_state_dict(state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')

    args = parser.parse_args()

    cfg = config.load_config(args.config)
#################3keyframe
    timestamps1_t0 = 0
    timestamps2_t0 = 1300
    timestamps3_t0 = 2600
    timestamps4_t0 = 3300
    stride = 1
    timestamps1 = [0, 46, 61, 74, 86, 98, 108, 122, 138, 155, 167, 178, 189, 202, 215, 231, 247, 264, 282, 301, 319,
                   334,347, 361, 375, 385, 395, 405, 421, 434, 446, 462, 477, 494, 504, 513, 528, 539, 555, 572, 587, 604,
                   621, 634, 649, 664, 680, 695, 703, 718, 726, 738, 746, 754, 769, 782, 797, 811, 822, 831, 841, 857,
                   865, 882, 897, 909, 921, 933, 948, 956, 971, 980, 995, 1007, 1016, 1030, 1038, 1053, 1061, 1069,
                   1081,1094, 1108, 1120, 1129, 1145, 1157, 1175, 1190, 1202, 1215, 1238, 1259, 1277, 1300, 1319, 1336, 1358,
                   1382, 1405, 1431, 1451, 1470, 1492, 1514, 1535, 1556, 1576, 1594, 1610, 1627, 1647, 1669, 1691]

    timestamps1 = [timestamp + timestamps1_t0 for timestamp in timestamps1[::stride]]

    dataset1 = ["/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/color/{:05d}.jpg".format(int(timestamp)) for timestamp in
                timestamps1]

    timestamps2 = [0., 9., 19., 27., 36., 48., 58., 82., 105., 131.,
                   151., 170., 192., 214., 235., 256., 276., 294., 310., 327.,
                   347., 369., 391., 417., 442., 461., 476., 495., 512., 527.,
                   546., 565., 588., 609., 627., 650., 673., 693., 719., 759.,
                   814., 839., 852., 868., 883., 892., 900., 904., 910., 917.,
                   925., 939., 955., 973., 984., 997., 1011., 1032., 1056., 1072.,
                   1093., 1103., 1118., 1127., 1134., 1142., 1150., 1158., 1171., 1180.,
                   1186., 1191., 1199., 1204., 1210., 1218., 1227., 1245., 1264., 1280.,
                   1291., 1314., 1331., 1340., 1359., 1369., 1378., 1393., 1410., 1429.,
                   1439., 1453., 1470., 1484., 1500., 1511., 1523., 1535., 1550., 1559.,
                   1576., 1592., 1604., 1616., 1628., 1643., 1658., 1673., 1691., 1697.]

    timestamps2 = [timestamp + timestamps2_t0 for timestamp in timestamps2[::stride]]

    dataset2 = ["/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/color/{:05d}.jpg".format(int(timestamp)) for timestamp in
                timestamps2]

    timestamps3 = [   0.,   11.,   22.,   30.,   38.,   46.,   56.,   68.,   84.,  101.,
         119.,  129.,  139.,  153.,  170.,  184.,  193.,  206.,  218.,  229.,
         242.,  259.,  276.,  292.,  304.,  316.,  328.,  343.,  358.,  373.,
         391.,  402.,  413.,  422.,  437.,  451.,  467.,  483.,  496.,  505.,
         519.,  534.,  549.,  562.,  577.,  587.,  596.,  608.,  619.,  634.,
         649.,  666.,  684.,  699.,  710.,  723.,  739.,  754.,  771.,  782.,
         792.,  806.,  824.,  839.,  854.,  862.,  878.,  888.,  896.,  916.,
         928.,  939.,  946.,  961.,  968.,  982.,  995., 1005., 1011., 1017.,
        1028., 1042., 1057., 1072., 1084., 1094., 1106., 1120., 1135., 1148.,
        1163., 1173., 1184., 1196., 1209., 1223., 1236., 1250., 1260., 1273.,
        1287., 1295.]

    timestamps3 = [timestamp + timestamps3_t0 for timestamp in timestamps3]

    dataset3 = ["/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/color/{:05d}.jpg".format(int(timestamp)) for timestamp in
                timestamps3]

    timestamps4 = [   0.,    5.,    9.,   16.,   23.,   31.,   39.,   54.,   71.,   82.,
          92.,  106.,  124.,  139.,  154.,  170.,  188.,  196.,  209.,  216.,
         228.,  239.,  246.,  261.,  268.,  282.,  295.,  305.,  311.,  317.,
         328.,  335.,  342.,  357.,  372.,  384.,  394.,  406.,  420.,  430.,
         441.,  455.,  463.,  473.,  484.,  496.,  509.,  523.,  536.,  550.,
         560.,  573.,  587.,  604.,  612.,  628.,  637.,  649.,  664.,  680.,
         688.,  705.,  714.,  730.,  743.,  756.,  774.,  784.,  794.,  804.,
         814.,  831.,  840.,  854.,  871.,  884.,  895.,  903.,  911.,  919.,
         927.,  936.,  947.,  958.,  965.,  972.,  979.,  992., 1005., 1020.,
        1033., 1041., 1056., 1074., 1088., 1104., 1112., 1120., 1128., 1137.,
        1145., 1152., 1157., 1162., 1166., 1170., 1172., 1174., 1178., 1182.,
        1188., 1195., 1205., 1213.]
    timestamps4 = [timestamp + timestamps4_t0 for timestamp in timestamps4]

    dataset4 = ["/data0/wjy/sgl/DATASET/Indoor/indoor_sq1/color/{:05d}.jpg".format(int(timestamp)) for timestamp in
                timestamps4]


################################cal
    net = DroidNet()
    load_pretrained(net, '/data0/wjy/sgl/GO-SLAM/pretrained/droid.pth')
    net.to('cuda:0').eval()
    loop_detector = Loop_detector(cfg, net, device='cuda:0')

    save_csv_path ='/data0/wjy/sgl/Co-SLAM-main-1/output/Indoor/indoor_sq1/corr_intrs.csv'
    min_flow_indices = loop_detector.find_minimum_flow(dataset1, dataset1, save_csv_path)
    min_timestamp_indices = (timestamps1[min_flow_indices[0]], timestamps1[min_flow_indices[1]])
    print(min_timestamp_indices)
