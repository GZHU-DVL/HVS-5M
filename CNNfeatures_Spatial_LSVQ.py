"""Extracting Spatial Features"""

import pandas as pd
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
        video_list = self.video_list[idx]  #other datasets
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
    parser = ArgumentParser(description='Extracting Spatial Features using Pre-Trained ConvNeXt')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='LSVQ', type=str, help='database name (default: KoNViD-1k)')
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


    if args.database == 'LSVQ':
        videos_dir = 'LSVQ/'
        features_dir = 'HVS-5M_LSVQ/SpatialFeature/'
        datainfo1 = 'data/labels_test_1080p.csv'
        datainfo2 = 'data/labels_train_test.csv'


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
    skip = [17157, 17158, 17159, 17162, 17163, 17165, 17167, 17168, 17169, 17170, 17177, 17179, 17181, 17182, 17184,  #Not in the LSVQ dataset
            17185, 17186, 17187, 17188, 17189, 17190, 17192, 17193, 17194, 17195, 17196, 17197, 17198, 17201, 17202,
            17204, 17205, 17208, 17209, 17211, 17212, 17214, 17215, 17216, 17217, 17218, 17221, 17222, 20008, 20009,
            20010, 20011, 20012, 20014, 20015, 38036, 38039, 38040, 38043, 38045, 38093, 38114, 38128, 38171, 38183,
            38184, 38218, 38242, 38245, 38289, 38290]

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



