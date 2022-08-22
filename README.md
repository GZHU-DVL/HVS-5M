# HVS-5M
HVS Revisited: A Comprehensive Video Quality Assessment  Framework
# Usage
## Requirment
* python==3.8.8
* torch==1.8.1
* torchvision==0.9.1
* torchsort==0.1.8
* detectron2==0.6
* scikit-video==1.1.11
* scikit-image==0.19.1
* scikit-learn==1.0.2
* scipy==1.8.0
* tensorboardX==2.4.1

## Dataset Preparation
**VQA Datasets.**

We test HVS-5M on six datasets, including [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [CVD2014](https://www.mv.helsinki.fi/home/msjnuuti/CVD2014/), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [LIVE-Qualcomm](http://live.ece.utexas.edu/research/incaptureDatabase/index.html), [YouTube-UGC](https://media.withyoutube.com/), and [LSVQ](https://github.com/baidut/PatchVQ), download the datasets from the official website. 

## Spatial Features
**The Content and edge features of the video are obtained by ConvNeXt.**

First, you need to download the dataset and copy the local address into the videos_dir of [CNNfeature_Spatial.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Spatial.py). Due to the particularity of the LSVQ dataset, we give a spatial feature version for extracting LSVQ in [CNNfeature_Spatial_LSVQ.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Spatial_LSVQ.py). In it, we marked the video sequence numbers that do not exist in the current version of LSVQ.

```
python CNNfeature_Spatial.py --database=database --frame_batch_size=16 \
python CNNfeature_Spatial_LSVQ.py --database=LSVQ --frame_batch_size=16
```

Please note that when extracting spatial features, you can choose the size of frame_batch_size according to your GPU. After running the [CNNfeature_Spatial.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Spatial.py) or [CNNfeature_Spatial_LSVQ.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Spatial_LSVQ.py), you can get the spatial features of each video in /HVS-5M_dataset/SpatialFeature/.


## Temporal Features
**The motion features of the video are obtained by SlowFast.**

First you need to download the SlowFast model into "./MotionExtractor/checkpoints/Kinetics/" 

[SlowFast]()

Similarly, for the other five datasets and LSVQ, we also give two versions to extract temporal features, namely [CNNfeatures_Temporal.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeatures_Temporal.py) and [CNNfeatures_Temporal_LSVQ.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeatures_Temporal_LSVQ.py).

```
python CNNfeature_Temporal.py --database=database --frame_batch_size=64 \
python CNNfeature_Temporal_LSVQ.py --database=LSVQ --frame_batch_size=64
```
Please note that frame_batch_size can only be 64 when extracting temporal features. After running the [CNNfeature_Temporal.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Temporal.py) or [CNNfeature_Temporal_LSVQ.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Temporal_LSVQ.py), you can get the temporal features of each video in /HVS-5M_dataset/TemporalFeature/.

## Fusion Features
**The spatial and temporal features are fused to obtain fusion features.**

```
python CNNfeature_Fusion.py --database=database --frame_batch_size=64 \
```

After running the [CNNfeature_Fusion.py](https://github.com/GZHU-DVL/HVS-5M/blob/main/CNNfeature_Fusion.py), you can get the fusion features of each video in /HVS-5M_dataset/.


