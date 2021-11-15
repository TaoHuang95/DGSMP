# Deep GSM prior for CASSI
This repository contains the Pytorch codes for paper **Deep Gaussian Scale Mixture Prior for Spectral Compressive Imaging** (***CVPR (2021)***) by [Tao Huang](https://github.com/TaoHuang95), [Weisheng Dong](https://see.xidian.edu.cn/faculty/wsdong/), [Xin Yuan](http://www.bell-labs.com/about/researcher-profiles/xyuan/).  
[[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.pdf)] [[arXiv](https://arxiv.org/pdf/2103.07152.pdf)] [[Project](https://see.xidian.edu.cn/faculty/wsdong/Projects/DGSM-SCI.htm)] [[supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Huang_Deep_Gaussian_Scale_CVPR_2021_supplemental.pdf)]



## Contents
1. [Overview](#Overview)
2. [Architecture](#Architecture)
3. [Usage](#Usage)
4. [Acknowledgements](#Acknowledgements)
5. [References](#References)
6. [Citation](#Citation)
7. [Contact](#Contact)

## Overview
We have proposed an interpretable hyperspectral image reconstruction method for coded aperture snapshot spectral
imaging. Different from existing works, our network was inspired by the Gaussian scale mixture (GSM) prior. Specifically,
the desired hyperspectral images were characterized by the GSM models and then the reconstruction problem was 
formulated as a MAP estimation problem. Instead of using a manually designed prior, we have proposed to learn the
scale prior of GSM by a DCNN. Furthermore, motivated by the auto-regressive model, the means of the GSM models
have been estimated as a weighted average of the spatial-spectral neighboring pixels, and these filter coefficients were
estimated by a DCNN as well aiming to learn sufficient spatial-spectral correlations of HSIs. Extensive 
experimental results on both synthetic and real datasets demonstrate that the proposed method outperforms 
existing state-of-the-art algorithms.

<p align="center">
<img src="/Images/Fig1.png" width="1200">
</p>
Fig. 1 A single shot measurement captured by [1] and 28 reconstructed spectral channels using our proposed method.

## Architecture
<p align="center">
<img src="/Images/Fig2.png" width="1200">
</p>
Fig. 2 Architecture of the proposed network for hyperspectral image reconstruction. The architectures of (a) the overall network, (b)
the measurement matrix, (c) the transposed version of the measurement matrix, (d) the weight generator, and (e) the filter generator.

## Usage
### Download the DGSMP repository
0. Requirements are Python 3 and PyTorch 1.2.0.
1. Download this repository via git
```
git clone https://github.com/TaoHuang95/DGSMP
```
or download the [zip file](https://github.com/TaoHuang95/DGSMP/archive/main.zip) manually.

### Download the training data
1. CAVE:28 channels (https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg) PW:ixoe
2. KAIST:28 channels (https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA) PW:5mmn


### Testing 
1. Testing on simulation data   
Run **Simulation/Test.py** to reconstruct 10 synthetic datasets ([Ziyi Meng](https://github.com/mengziyi64/TSA-Net)). The results will be saved in 'Simulation/Results/' in the MatFile format.  
2. Testing on real data   
Run **Real/Test.py** to reconstruct 5 real datasets ([Ziyi Meng](https://github.com/mengziyi64/TSA-Net)). The results will be saved in 'Real/Results/' in the MatFile format.  

### Training 
1. Training simulation model
    1) Put hyperspectral datasets (Ground truth) into corrsponding path, i.e., 'Simulation/Data/Training_data/'.
    2) Run **Simulation/Train.py**.
2. Training real data model  
    1) Put hyperspectral datasets (Ground truth) into corrsponding path, i.e., 'Real/Data/Training_data/'.  
    2) Run **Real/Train.py**.

## Acknowledgements
We thank the author of TSA-Net[1] ([Ziyi Meng](https://github.com/mengziyi64/TSA-Net)) for providing simulation data and real data.

## References
[1] Ziyi Meng, Jiawei Ma, and Xin Yuan. End-to-end low cost compressive spectral imaging with spatial-spectral self-attention. In Proceedings of the European Conference on
Computer Vision (ECCV), August 2020.

## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@InProceedings{Huang_2021_CVPR,
    author    = {Huang, Tao and Dong, Weisheng and Yuan, Xin and Wu, Jinjian and Shi, Guangming},
    title     = {Deep Gaussian Scale Mixture Prior for Spectral Compressive Imaging},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16216-16225}
}
```

## Contact
Tao Huang, Xidian University, Email: thuang_666@stu.xidian.edu.cn, thuang951223@163.com  
Weisheng Dong, Xidian University, Email: wsdong@mail.xidian.edu.cn  
Xin Yuan, Bell Labs, Email: xin_x.yuan@nokia-bell-labs.com
