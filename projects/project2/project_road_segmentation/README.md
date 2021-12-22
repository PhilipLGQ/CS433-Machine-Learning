# Machine Learning Project 2 (2021): Aerial Road Segmentation

## Introduction
In this project, we train an equal-weight ensemble U-Net classifier to segment road regions from real satellite images cropped from Google Map, we also implement a standard SegNet model and compare our models to single segmentation networks. The training set includes 100 satellite images of size 400x400, each corresponding to a binary-color mask image showing the groundtruth road region. After the training process, we generate prediction masks on 50 test images of size 608x608. These masks are then cropped into small patches of size 16x16 and are transformed into a .csv file with patch-mean based thresholding (road=1, background=0). 

## Getting Started
* Provided scripts and model training are built and tested under Anaconda virtual environment with Python 3.7.11. 
* In order to reproduce our best submission result to AICrowd with pretrained models. Please make sure the packages in `requirements.txt` are properly installed.
```bash
pip install -r requirements.txt
```
* To train from scratch, please download the dataset from [AICrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files) and extract them under both `data/` and  `data_segnet/` directory. Only test images are required if you use pretrained models to generate the result. 
* Download our pretrained models from [here](), please make sure they are placed under `weights/`. 

## Files
### `run.py`
Generates our best submission (.csv file) with pretrained models (all 3 U-Net models needed). Generated masks and the csv submission file will be restored under `pred_imgs/` and `submission/` respectively.

### `dataset_UNet.py`
Includes helper functions for U-Net data loading, data augmentation, data rescaling, and data saving.

### `dataset_SegNet.py`
Includes helper functions for SegNet data loading, data augmentation, data rescaling, and data saving.

### `model/`
* **`SegNet.py`**: Standard SegNet model, with encoder blocks and decoder blocks
* **`UNet.py`**: Overfitting optimized U-Net model, add dropout and batch normalizaation layers after "deconvolution".
* **`dilated_UNet`**: U-Net model with parallel dilated convolution module at network "bottom", inspired by [[1]](#1).

### `train_UNet`
Train U-Net models from scratch, please make sure training data is extracted under `data/`.

### `train_SegNet`
Train SegNet model from scratch, please make sure training data is extracted under `data_segnet/`.

### `mask_to_submission`
Includes helper functions for generating the submission csv file.


## Best Model Performance
* **`Model`**: Equal-Weight Ensemble U-Net
* **`Submission ID`**: , **`Submission Username`**:
* Performance: **`F1 Score`**: 0.903, **`Accuracy`**: 0.949


## Authors
* *Guanqun Liu*
* [*Xianjie Dai*](https://github.com/xianjiedai)
* [*Yixuan Xu*](https://github.com/Alvorecer721)

## References
<a id="1">[1]</a>
S. Piao and J. Liu, “Accuracy improvement of unet based on dilated convolution,” Journal of Physics: Conference Series, vol. 1345, no. 5, p. 052066, 2019. 
