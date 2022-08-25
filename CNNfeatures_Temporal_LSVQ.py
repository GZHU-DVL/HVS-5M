"""Extracting Video Temporal Features"""

from argparse import ArgumentParser
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
import h5py
import numpy as np
import random
import time
import os
import pandas as pd
from pathlib import Path
class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx] + '.mp4'
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        print('video_width: {} video_height: {}'.format(video_width, video_height))
        sample = {'video': video_data, 'score': video_score}
        return sample

class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self):
        super(CNNModel, self).__init__()
        from MotionExtractor.get_motionextractor_model import make_motion_model
        model = make_motion_model()
        self.features = model

    def forward(self, x):
        motion_feature_maps = self.features(x)
        motion_features_mean = nn.functional.adaptive_avg_pool2d(motion_feature_maps[1], 1)  #  motion_feature_maps--->motion_features
        motion_features_std = global_std_pool3d(motion_feature_maps[1])
        motion_features_mean = torch.squeeze(motion_features_mean).permute(1, 0)
        motion_features_std = torch.squeeze(motion_features_std).permute(1, 0)
        return motion_features_mean, motion_features_std

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)

def global_std_pool3d(x):
    """3D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], x.size()[2], -1, 1), dim=3, keepdim=True)

from MotionExtractor.slowfast.visualization.utils import process_cv2_inputs
from MotionExtractor.slowfast.utils.parser import load_config, parse_args
def get_features(video_data, frame_batch_size=64, device='cuda'):
    """motion feature extraction"""
    extractor = CNNModel().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    motion_features1 = torch.Tensor().to(device)
    motion_features2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        args = parse_args()
        cfg = load_config(args)
        if video_length <= frame_batch_size:
            batch = video_data[0:video_length]
            inputs = process_cv2_inputs(batch, cfg)
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda()

            motion_features_mean, motion_features_std = extractor(inputs)
            motion_features1 = torch.cat((motion_features1, motion_features_mean), 0)
            motion_features2 = torch.cat((motion_features2, motion_features_std), 0)
            Temporal_features = torch.cat((motion_features1, motion_features2), 1).squeeze()
        else:
            num_block = 0
            while frame_end < video_length:
                batch = video_data[frame_start:frame_end]
                inputs = process_cv2_inputs(batch, cfg)
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda()

                motion_features_mean, motion_features_std = extractor(inputs)  #extract motion features
                motion_features1 = torch.cat((motion_features1, motion_features_mean), 0)
                motion_features2 = torch.cat((motion_features2, motion_features_std), 0)
                frame_end += frame_batch_size
                frame_start += frame_batch_size
                num_block = num_block + 1

            last_batch = video_data[(video_length - frame_batch_size):video_length]
            inputs = process_cv2_inputs(last_batch, cfg)
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda()
            motion_features_mean, motion_features_std = extractor(inputs)
            index = torch.linspace(0, (frame_batch_size - 1), 32).long()
            last_batch_index = (video_length - frame_batch_size) + index
            elements = torch.where(last_batch_index >= frame_batch_size * num_block)
            motion_features1 = torch.cat((motion_features1, motion_features_mean[elements[0], :]), 0)
            motion_features2 = torch.cat((motion_features2, motion_features_std[elements[0], :]), 0)
            Temporal_features = torch.cat((motion_features1, motion_features2), 1).squeeze()

    if Temporal_features.ndim == 1:
        Temporal_features = Temporal_features.unsqueeze(0)

    return Temporal_features

if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting Temporal Features using Pre-Trained SlowFast')
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument('--database', default='LSVQ', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument("--ith", type=int, default=0, help='start video id')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'LSVQ':
        videos_dir = 'LSVQ/'
        features_dir = 'HVS-5M_LSVQ/SpatialFeature/'
        datainfo1 = 'data/labels_train_test.csv'
        datainfo2 = 'data/labels_test_1080p.csv'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    Info1 = pd.read_csv(datainfo1, encoding='gbk', nrows=17223)
    Info2 = pd.read_csv(datainfo2, encoding='gbk', nrows=2794)
    Info3 = pd.read_csv(datainfo1, encoding='gbk', skiprows=17223)
    Info3.columns = ['name', 'height', 'width', 'mos', 'frame_number']
    Info4 = pd.read_csv(datainfo2, encoding='gbk', skiprows=2794)
    Info4.columns = ['name', 'p1', 'p2', 'p3', 'height', 'width', 'mos_p1', 'mos_p2', 'mos_p3', 'mos', 'frame_number',
                     'fn_last_frame', 'left_p1', 'right_p1', 'top_p1', 'bottom_p1', 'start_p1', 'end_p1', 'left_p2',
                     'right_p2', 'top_p2', 'bottom_p2', 'start_p2', 'end_p2', 'left_p3', 'right_p3', 'top_p3',
                     'bottom_p3', 'start_p3', 'end_p3', 'top_vid', 'left_vid', 'bottom_vid', 'right_vid', 'start_vid',
                     'end_vid', 'is_valid']

    merge = [Info1, Info2, Info3, Info4]
    Info = pd.concat(merge, ignore_index=True)
    mos = Info['mos']
    width = Info['width']
    height = Info['height']
    video_list = Info['name']
    video_format = 'RGB'

    dataset = VideoDataset(videos_dir, video_list, mos, video_format, width, height)
    skip = [17157, 17158, 17159, 17162, 17163, 17165, 17167, 17168, 17169, 17170, 17177, 17179, 17181, 17182, 17184,  # Not in the LSVQ dataset
            17185, 17186, 17187, 17188, 17189, 17190, 17192, 17193, 17194, 17195, 17196, 17197, 17198, 17201, 17202,
            17204, 17205, 17208, 17209, 17211, 17212, 17214, 17215, 17216, 17217, 17218, 17221, 17222, 20008, 20009,
            20010, 20011, 20012, 20014, 20015, 38036, 38039, 38040, 38043, 38045, 38093, 38114, 38128, 38171, 38183,
            38184, 38218, 38242, 38245, 38289, 38290]

    max_len = 0
    min_len = 100000
    for i in range(args.ith, len(dataset)):
        if i not in skip:
            start = time.time()
            current_data = dataset[i]
            print('Video {}: length {}'.format(i, current_data['video'].shape[0]))
            if max_len < current_data['video'].shape[0]:
                max_len = current_data['video'].shape[0]
            if min_len > current_data['video'].shape[0]:
                min_len = current_data['video'].shape[0]
            Temporal_features = get_features(current_data['video'], args.frame_batch_size, device)
            print(Temporal_features.shape)
            np.save(features_dir + str(i) + '_Temporal', Temporal_features.to('cpu').numpy())
            np.save(features_dir + str(i) + '_score', current_data['score'])
            end = time.time()
            print('{} seconds'.format(end - start))
    print('Max length: {} Min length: {}'.format(max_len, min_len))
