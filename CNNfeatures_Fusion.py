"""Concate the spatial and temporal features"""

from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import time
import os
def fuse_features(spatial_feature, temporal_feature, frame_batch_size=64, device='cuda'):
    video_length = spatial_feature.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    fusion_features = np.empty(shape=[0, 8192], dtype=np.float32)

    if video_length <= frame_batch_size:
        index = torch.linspace(0, (spatial_feature.shape[0] - 1), 32).long()
        spatial_feature = torch.from_numpy(spatial_feature)
        fusion_features = torch.index_select(spatial_feature, 0, index)

    else:
        index = torch.linspace(0, (frame_batch_size - 1), 32).long().numpy()
        num_block = 0
        while frame_end < video_length:
            batch = spatial_feature[frame_start:frame_end, :]
            fusion_features = np.concatenate((fusion_features, batch[index, :]), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size
            num_block = num_block + 1

        last_batch_index = (video_length - frame_batch_size) + index
        elements = np.where(last_batch_index >= frame_batch_size * num_block)
        elements = elements[0] + (video_length - frame_batch_size)
        fusion_features = np.concatenate((fusion_features, spatial_feature[elements, :]), 0)
    fusion_features = np.concatenate((fusion_features, temporal_feature), 1)
    return fusion_features

if __name__ == "__main__":
    parser = ArgumentParser(description='Video Feature Fusion')
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
        features_dir = 'HVS-5M_KoNViD-1k/'
        datainfo = 'data/KoNViD-1kinfo.mat'
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/'
        features_dir = 'HVS-5M_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/'
        features_dir = 'HVS-5M_LIVE-VQC/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/'
        features_dir = 'HVS-5M_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'YouTube-UGC':
        videos_dir = 'YouTube-UGC/'
        features_dir = 'HVS-5M_YouTube-UGC/'
        datainfo = 'data/YOUTUBE_UGC_metadata.csv'
    if args.database == 'LSVQ':
        videos_dir = 'LSVQ/'
        features_dir = 'HVS-5M_LSVQ/'
        datainfo1 = 'data/labels_train_test.csv'
        datainfo2 = 'data/labels_test_1080p.csv'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    for i in range(args.ith, len(video_names)):

        start = time.time()
        print('Video: {}'.format(i))
        spatial_features = np.load(features_dir+ 'SpatialFeature/' + str(i) + '_' + 'Spatial' + '.npy')
        print(spatial_features.shape)
        motion_features = np.load(features_dir + 'TemporalFeature/' +str(i)+ '_' + 'Temporal' + '.npy')
        print(motion_features.shape)
        score = np.load(features_dir + 'TemporalFeature/' + str(i) + '_score.npy')
        fusion_features = fuse_features(spatial_features, motion_features, args.frame_batch_size)
        print(fusion_features.shape)
        np.save(features_dir + str(i) + '_' + 'fusion_features', fusion_features)
        np.save(features_dir + str(i) + '_score', score)
        end = time.time()
        print('{} seconds'.format(end-start))

