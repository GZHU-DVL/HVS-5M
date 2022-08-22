"""Extracting Spatial Features"""

import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
from SAMNet.SAMNet import FastSal as net
import time
from ConvNeXt.convnext import convnext_xlarge
class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, videos_dir, video_list, mos, video_format='RGB', width=None, height=None):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_list = video_list
        self.mos = mos
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        canny = torch.Tensor()
        if self.video_list[idx][11] == '_':  # KoNViD-1k only
            video_list = self.video_list[idx][0:11] + '.mp4'
        elif self.video_list[idx][10] == '_':
            video_list = self.video_list[idx][0:10] + '.mp4'
        else:
            video_list = self.video_list[idx][0:9] + '.mp4'
        #video_list = self.video_list[idx]  #other datasets
        print(video_list)
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_list), self.height, self.width,
                                          inputdict={'-pix_fmt': 'yuvj420p'})  #load video_data
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_list)) #load video_data
        model = net()
        state_dict = torch.load(args.pretrained, map_location=torch.device('cuda'))
        if list(state_dict.keys())[0][:7] == 'module.':
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)  #load SAMNet
        model = model.cuda()
        model.eval()
        mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
        std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
        saliency_map = torch.Tensor()
        saliency_map = saliency_map.to(device)
        for i in range(video_data.shape[0]):
            video =  np.array(video_data[i],dtype=np.float32)
            video = (video - mean) / std
            video = video.transpose((2, 0, 1))
            video = torch.from_numpy(video).unsqueeze(0)
            video = video.to(device)
            with torch.no_grad():
                pred = model(video)[:, 0, :, :].unsqueeze(1)  #saliency map
            assert pred.shape[-2:] == video.shape[-2:], '%s vs. %s' % (str(pred.shape), str(video.shape))
            pred = pred.squeeze(1)
            pred = (torch.ceil(pred[0] * 255)).cpu().numpy()
            pred = np.where(pred >= args.h, pred + 350, pred)
            pred = np.where(pred < args.h, pred + 250, pred)
            newimg_h = int(video_data[i].shape[0]/32)  #resize saliency map
            newimg_w = int(video_data[i].shape[1]/32)
            pred = cv2.resize(pred, (newimg_w, newimg_h))
            pred = (pred / 255).astype(np.float32)
            pred = torch.from_numpy(pred)
            pred = pred.unsqueeze(0)
            pred = pred.unsqueeze(0)
            (B, G, R) = cv2.split(video_data[i])
            B_middle = cv2.Canny(B, args.l, args.u)    #detect edge map
            G_middle = cv2.Canny(G, args.l, args.u)
            R_middle = cv2.Canny(R, args.l, args.u)
            RGB_middle = np.dstack((B_middle, G_middle, R_middle))  #concate edge map
            RGB_middle = torch.from_numpy(RGB_middle)
            RGB_middle = RGB_middle.unsqueeze(0)
            canny = torch.cat((canny, RGB_middle), 0)
            if i == 0:
                saliency_map = pred
            else:
                saliency_map = torch.cat((saliency_map, pred), 0)
        canny = canny.cpu().numpy()
        canny = np.asarray(canny)
        video_data = np.asarray(video_data)
        video_mos = self.mos[idx]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        content = torch.zeros([video_length, video_channel, video_height, video_width])
        edge_maps = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame_video = video_data[frame_idx]
            frame_canny = canny[frame_idx]
            frame_video = Image.fromarray(frame_video)
            frame_canny = Image.fromarray(np.uint8(frame_canny))
            frame_video = transform(frame_video)
            frame_canny = transform(frame_canny)
            content[frame_idx] = frame_video
            edge_maps[frame_idx] = frame_canny
        sample = {'content': content,
                  'mos': video_mos,
                  'edge_maps': edge_maps,
                  'saliency_map':saliency_map}
        return sample



def get_features(current_content, current_edge_maps, current_saliency_map,frame_batch_size=16, device='cuda'):
    """feature extraction"""
    extractor = convnext_xlarge(pretrained=True,in_22k = True).to(device)
    video_length = current_content.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    content1 = torch.Tensor().to(device)
    content2 = torch.Tensor().to(device)
    edge1 = torch.Tensor().to(device)
    edge2 = torch.Tensor().to(device)
    extractor.eval()

    with torch.no_grad():
        while frame_end < video_length:
            batch_content = current_content[frame_start:frame_end].to(device)
            batch_edge_maps = current_edge_maps[frame_start:frame_end].to(device)
            batch_saliency_map = current_saliency_map[frame_start:frame_end].to(device)
            content_mean, content_std, edge_mean, edge_std = extractor(batch_content, batch_edge_maps,batch_saliency_map) #extract (content/edge) features
            content1 = torch.cat((content1, content_mean), 0).to(device)
            content2 = torch.cat((content2, content_std), 0).to(device)
            edge1 = torch.cat((edge1, edge_mean), 0).to(device)
            edge2 = torch.cat((edge2, edge_std), 0).to(device)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_content = current_content[frame_start:video_length].to(device)
        last_edge_maps = current_edge_maps[frame_start:frame_end].to(device)
        last_saliency_map = current_saliency_map[frame_start:frame_end].to(device)
        content_mean, content_std, edge_mean, edge_std = extractor(last_content, last_edge_maps,last_saliency_map)  #extract (content/edge) features
        content1 = torch.cat((content1, content_mean), 0).to(device)
        content2 = torch.cat((content2, content_std), 0).to(device)
        edge1 = torch.cat((edge1, edge_mean), 0).to(device)
        edge2 = torch.cat((edge2, edge_std), 0).to(device)
        Spatial_feature = torch.cat((content1, content2, edge1, edge2), 1).squeeze().to(device)

    return Spatial_feature


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='KoNViD-1k', type=str, help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=16, help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument('--pretrained', default='./Pretrained/SAMNet_with_ImageNet_pretrain.pth', type=str, help='pretrained model')
    parser.add_argument('--h', default=100, type=int, help='threshold of saliency map')
    parser.add_argument('--u', default=140, type=int, help='upper threshold of Canny')
    parser.add_argument('--l', default=5, type=int, help='lower threshold of Canny')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = 'KoNViD-1k/'  # videos dir
        features_dir = 'HVS-5M_KoNViD-1k/SpatialFeature/'
        datainfo = 'data/KoNViD-1kinfo.mat'
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/'
        features_dir = 'HVS-5M_CVD2014/SpatialFeature/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/'
        features_dir = 'HVS-5M_LIVE-VQC/SpatialFeature/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/'
        features_dir = 'HVS-5M_LIVE-Qualcomm/SpatialFeature/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if args.database == 'YouTube-UGC':
        videos_dir = 'YouTube-UGC/'
        features_dir = 'HVS-5M_YouTube-UGC/SpatialFeature/'
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

    for i in range(len(dataset)):
        start = time.time()
        current_data = dataset[i]
        current_saliency_map = current_data['saliency_map']
        current_edge_maps = current_data['edge_maps']
        current_content = current_data['content']
        current_mos = current_data['mos']
        print('Video {}: length {}'.format(i, current_content.shape[0]))
        Spatial_features = get_features(current_content, current_edge_maps, current_saliency_map, args.frame_batch_size, device) #Spatial_features
        print(Spatial_features.shape)
        np.save(features_dir + str(i) + '_Spatial', Spatial_features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_mos)
        end = time.time()
        print('{} seconds'.format(end - start))



