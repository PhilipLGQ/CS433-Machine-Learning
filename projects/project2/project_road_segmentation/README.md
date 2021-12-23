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
* Download our pretrained models from [UNet here](https://drive.google.com/file/d/1kMt53JgvZkGScvjtHSgpJbElNd7sSDTX/view?usp=sharing) and [SegNet here](https://drive.google.com/file/d/14LWOf0vvGJZuv97c1HWh49lH1ZdspYZX/view?usp=sharing), please make sure UNet models are placed under `weights/` and the SegNet model is placed under `checkpoint/Segnet/model`. Execute directly in Python IDE (e.g. PyCharm) or run the following to get the submission csv and predicted masks (UNet):
```bash
python run.py
```
* If you want to check the predicted masks of SegNet with pretrained model, please execute
```bash
python SegNet_predict.py
```
* If you want to train from scratch, please execute the following to train all UNet models and the SegNet model:
```bash
python train_UNet.py
python generate_Seg.py
python train_Seg.py
```
then execute `run.py` and `SegNet_predict.py` to obtain the prediction csv file under `submission/`, prediction masks of UNet: `data/prediction/`, prediction masks of SegNet `data_segnet/prediction/`.


## File & Directory
### `run.py`
Generates our best submission (.csv file) with pretrained models (all 3 U-Net models needed). Generated masks and the csv submission file will be restored under `data/prediction/` and `submission/` respectively. Please make sure testing data is extracted under `data/`.

### `dataset_UNet.py`
Includes helper functions for U-Net data loading, data augmentation, data rescaling, and data saving.

### `dataset_SegNet.py`
Includes helper functions for SegNet data loading, data augmentation, data rescaling, and data saving.

### `metric_loss.py`
Includes functions for F1-score calculation and dice loss calculation.

### `train_UNet`
Train U-Net models from scratch, please make sure training data is extracted under `data/`.

### `generate_Seg.py`
Generate training set and test set for SegNet training under `data_segnet/`

### `train_SegNet.py`
Train SegNet model from scratch, please make sure training data is extracted under `data_segnet/`.

### `SegNet_predict.py`
Generate and save predicted masks of SegNet, please make sure testing data is extracted under `data_segnet/`

### `mask_to_submission.py`
Includes helper functions for generating the submission csv file.

### `model/`
* **`SegNet.py`**: Standard SegNet model, with encoder blocks and decoder blocks
* **`UNet.py`**: Overfitting optimized U-Net model, add dropout and batch normalizaation layers after "deconvolution".
* **`dilated_UNet`**: U-Net model with parallel dilated convolution module at network "bottom", inspired by [[1]](#1).

### `data/`, `data_segnet/`
Restores training data and test data for UNet and SegNet respectively. 

### `checkpoint/`
* **`model/`**: Save SegNet checkpoint models at each epoch. Please make sure the SegNet pretrained model is placed here to generate result directly.
* SegNet checkpoint output (predicted masks according to current weights) will generate here.

### `weights/`
Save UNet models here. Please make sure the UNet pretrained models are placed here to generate result directly

### `submission/`
Generates submission csv file here. 

## Best Model Performance
* **`Model`**: Equal-Weight Ensemble U-Net
* **`Submission ID`**: xianjiedai , **`Submission Username`**: 169451
* Performance: **`F1 Score`**: 0.903, **`Accuracy`**: 0.949

## Authors
* *Guanqun Liu*
* [*Xianjie Dai*](https://github.com/xianjiedai)
* [*Yixuan Xu*](https://github.com/Alvorecer721)

## References
<a id="1">[1]</a>
S. Piao and J. Liu, “Accuracy improvement of unet based on dilated convolution,” Journal of Physics: Conference Series, vol. 1345, no. 5, p. 052066, 2019. 
