# HVS-5M
HVS Revisited: A Comprehensive Video Quality Assessment  Framework
## Usage
### Requirment
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

### Dataset Preparation
**VQA Datasets.**

We test HVS-5M on six datasets, including [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [CVD2014](https://www.mv.helsinki.fi/home/msjnuuti/CVD2014/), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [LIVE-Qualcomm](http://live.ece.utexas.edu/research/incaptureDatabase/index.html), [YouTube-UGC](https://media.withyoutube.com/), and [LSVQ](https://github.com/baidut/PatchVQ), download the datasets from the official website. After downloading the dataset, copy the local address into the videos_dir of CNNfeature_Spatial.py.
