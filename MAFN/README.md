# Hyperspectral Image Classification with Multi-attention Fusion Network

This repository implementates 7 frameworks for hyperspectral image classification based on Keras and PyTorch.

Some of our code references the projects
* [Spectral-spatial residualnetwork for hyperspectral image classification: A 3-D deep learning framework](https://github.com/zilongzhong/SSRN)
* [A Fast Dense Spectral-Spatial Convolution Network Framework for Hyperspectral Images Classification](https://github.com/shuguang-52/FDSSC.git) 

Requirements：
------- 
```
CUDA = 9.0
python>=3.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
tensorflow-gpu = 1.8.0
keras>=2.2.0
numpy>=1.16.0
```

Datasets:
------- 
You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./datasets` folder.

Usage:
-------
Take KSC dataset as an example: 
1. Download the required dataset and add the corresponding path in the file `./KSC_data.py`
2. Taking the MAFN framework as an example, run `./KSC/main.py`. 

Paper:
------- 
* [Two-CNN](https://ieeexplore.ieee.org/document/7927776)
* [BAM-CM](https://arxiv.org/pdf/1906.04379.pdf)
* [SSAN](https://ieeexplore.ieee.org/document/8909379)
* [SSRN](https://ieeexplore.ieee.org/document/8061020)
* [DFFN](https://ieeexplore.ieee.org/document/8283837)
* [DBDA](https://www.researchgate.net/publication/339192574_Classification_of_Hyperspectral_Image_Based_on_Double-Branch_Dual-Attention_Mechanism_Network)
* [MAFN]![image](Fig1.png)
Fig1. The structure of the MAFN network. MAFN consists of three main components: spectral feature extraction, spatial feature extraction and joint spectral-spatial feature extraction. MAFN employs Band Attention Module (BAM) and Spatial Attention Module (SAM) respectively to alleviate the influence of redundant bands and interfering pixels. MAFN realizes feature reuse and obtains complementary information from different levels by combining multi-attention and multi-level fusion mechanisms, which can extract more representative features.
