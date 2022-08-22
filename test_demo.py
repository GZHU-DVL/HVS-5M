import torch
from torchvision import transforms
import skvideo.io
from PIL import Image
from CNNfeatures_Spatial import get_features as get_spatial_features
from CNNfeatures_Temporal import get_features as get_temporal_features
from CNNfeatures_Fusion import fuse_features
from VQAmodel import VQAModel
from argparse import ArgumentParser
import time
import numpy as np
import cv2
from SAMNet.SAMNet import FastSal as net
if __name__ == "__main__":
    parser = ArgumentParser(description='"Test Demo of HVS-5M')
    parser.add_argument('--model_path', default="models/HVS-5M_K", type=str)
    parser.add_argument('--video_path', default='data/test.mp4', type=str, help='video path (default: data/test.mp4)')
    parser.add_argument('--video_format', default='RGB', type=str, help='video format: RGB or YUV420 (default: RGB)')
    parser.add_argument('--video_width', type=int, default=None, help='video width')
    parser.add_argument('--video_height', type=int, default=None, help='video height')
    parser.add_argument('--frame_batch_size', type=int, default=64, help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['K'], help='C K L N Y Q')
    parser.add_argument('--pretrained', default='./Pretrained/SAMNet_with_ImageNet_pretrain.pth', type=str, help='pretrained model')
    parser.add_argument('--h', default=100, type=int, help='threshold of saliency map')
    parser.add_argument('--u', default=140, type=int, help='upper threshold of Canny')
    parser.add_argument('--l', default=5, type=int, help='lower threshold of Canny')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    # data preparation
    assert args.video_format == 'YUV420' or args.video_format == 'RGB'
    if args.video_format == 'YUV420':
        video_data = skvideo.io.vread(args.video_path, args.video_height, args.video_width, inputdict={'-pix_fmt': 'yuvj420p'})
    else:
        video_data = skvideo.io.vread(args.video_path)
    canny = torch.Tensor()
    model = net()
    state_dict = torch.load(args.pretrained, map_location=torch.device('cuda'))
    if list(state_dict.keys())[0][:7] == 'module.':
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)  # load SAMNet
    model = model.cuda()
    model.eval()
    mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
    std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)
    saliency_map = torch.Tensor()
    saliency_map = saliency_map.to(device)
    for i in range(video_data.shape[0]):
        video = np.array(video_data[i], dtype=np.float32)
        video = (video - mean) / std
        video = video.transpose((2, 0, 1))
        video = torch.from_numpy(video).unsqueeze(0)
        video = video.to(device)
        with torch.no_grad():
            pred = model(video)[:, 0, :, :].unsqueeze(1)  # saliency map
        assert pred.shape[-2:] == video.shape[-2:], '%s vs. %s' % (str(pred.shape), str(video.shape))
        pred = pred.squeeze(1)
        pred = (torch.ceil(pred[0] * 255)).cpu().numpy()
        pred = np.where(pred >= args.h, pred + 350, pred)
        pred = np.where(pred < args.h, pred + 250, pred)
        newimg_h = int(video_data[i].shape[0] / 32)  # resize saliency map
        newimg_w = int(video_data[i].shape[1] / 32)
        pred = cv2.resize(pred, (newimg_w, newimg_h))
        pred = (pred / 255).astype(np.float32)
        pred = torch.from_numpy(pred)
        pred = pred.unsqueeze(0)
        pred = pred.unsqueeze(0)
        (B, G, R) = cv2.split(video_data[i])
        B_middle = cv2.Canny(B, args.l, args.u)  # edge map
        G_middle = cv2.Canny(G, args.l, args.u)
        R_middle = cv2.Canny(R, args.l, args.u)
        RGB_middle = np.dstack((B_middle, G_middle, R_middle))  # concate edge map
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

    # fusion feature extraction
    spatial_features = get_spatial_features(content, edge_maps, saliency_map, frame_batch_size=16, device=device)
    #print(spatial_features.shape)
    temporal_features = get_temporal_features(video_data, frame_batch_size=args.frame_batch_size, device=device)
    #print(temporal_features.shape)
    fusion_features = fuse_features(spatial_features.to('cpu').numpy(), temporal_features.to('cpu').numpy(), args.frame_batch_size)
    fusion_features = torch.from_numpy(fusion_features).cuda()
    fusion_features = torch.unsqueeze(fusion_features, 0)
    print(fusion_features.shape)

    # database initial
    scale = dict()
    m = dict()
    for dataset in args.trained_datasets:
        scale[dataset] = 1
        m[dataset] = 0

    # quality prediction
    model = VQAModel(scale=scale, m=m).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.no_grad():
        input_length = fusion_features.shape[1] * torch.ones(1, 1, dtype=torch.long)
        relative_score, mapped_score, aligned_score = model([(fusion_features, input_length, args.trained_datasets)])
        y_pred = mapped_score[0][0].to('cpu').numpy()
        print("Predicted perceptual quality: {}".format(y_pred))

    end = time.time()

    print('Time: {} s'.format(end-start))