# Machine Learning Project 2 (2021): Aerial Road Segmentation

## Introduction
In this project, we train an equal-weight concatenated U-Net classifier to segment road regions from real satellite images cropped from Google Map. Besides, we also implement several single segmentation networks to compare the performance. The training set includes 100 satellite images of size 400x400, each corresponding to a binary-color mask image showing the groundtruth road region. After the training process, we generate prediction masks on 50 test images of size 608x608. These masks are then cropped into small patches of size 16x16 and are transformed into a .csv file with patch-mean based thresholding (road=1, background=0). 

## Getting Started
Provided scripts and model training are built and tested under Anaconda virtual environment with Python 3.7.11. In order to reproduce our best submission result to AICrowd with pretrained models. Please make sure the packages in `requirements.txt` are properly installed.

```bash
pip install -r requirements.txt
```


Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
 
 
## Running Prerequisites


## Implementation Details


## Model Performance
Our best model: Ridge Regression with imputation through k-means clustering, test accuracy: 0.803, F1 score: 0.685


## Authors
* *Guanqun Liu*
* [*Xianjie Dai*](https://github.com/xianjiedai)
* [*Yixuan Xu*](https://github.com/Alvorecer721)
