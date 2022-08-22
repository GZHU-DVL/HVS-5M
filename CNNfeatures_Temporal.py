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
        if self.video_names[idx][11] == '_':  # KoNViD-1k only
            video_name = self.video_names[idx][0:11] + '.mp4'
        elif self.video_names[idx][10] == '_':
            video_name = self.video_names[idx][0:10] + '.mp4'
        else:
            video_name = self.video_names[idx][0:9] + '.mp4'
        #video_name = self.video_names[idx]  #other datasets

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
    parser.add_argument('--database', default='KoNViD-1k', type=str,
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

    if args.database == 'KoNViD-1k':
        videos_dir = 'KoNViD-1k/'
        features_dir = 'HVS-5M_KoNViD-1k/TemporalFeature/'
        datainfo = 'data/KoNViD-1kinfo.mat'
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/'
        features_dir = 'HVS-5M_CVD2014/TemporalFeature/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/'
        features_dir = 'HVS-5M_LIVE-VQC/TemporalFeature/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/'
        features_dir = 'HVS-5M_LIVE-Qualcomm/TemporalFeature/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'YouTube-UGC':
        videos_dir = 'YouTube-UGC/'
        features_dir = 'HVS-5M_YouTube-UGC/TemporalFeature/'
        datainfo = 'data/YOUTUBE_UGC_metadata.csv'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)

    max_len = 0
    min_len = 100000
    for i in range(args.ith, len(dataset)):
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
