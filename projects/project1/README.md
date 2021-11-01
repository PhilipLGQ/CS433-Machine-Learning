# Machine Learning Project 1 (2021): The Higgs Boson Machine Learning Challenge

The "Higgs Boson" is an elementary particle belonging to the Standard Model of particle physics. It is produced by exerting the quantum excitation of the Higgs field. Due to its rapidness of decay, scientists found it difficult to directly observe its decay signal due to background noise. The goal of this machine learning challenge is to improve the discovery significance of the particle by classifying provided data events into "tau tau decay of Higgs Boson" or "background". The aim of this project is to realize a satisfying performance on binary classification.

Detailed project outlines and requirements can be found in the [project description](./project1_description.pdf). This challenge includes an [AIcrowd online contest] (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), the original kaggle contest is [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson) (2014).

## Getting Started
Provided scripts and notebook files were built and tested with a conda environment with python version 3.7.11. 
The following external libraries are used within the scripts:

```bash
numpy (as np)
matplotlib.pyplot (as plt)
```

## Running Prerequisites
Before running the scripts and notebook files, you should keep the folder structure under folder **scripts** as follows:

```bash
  .
  ├── costs.py
  ├── cross_validation.py
  ├── data
  │   ├── test.csv               # training set, extract first
  │   └── train.csv              # test set, extract first
  ├── helpers.py
  ├── implementations.py
  ├── preprocess.py
  ├── proj1_helpers.py
  ├── project1.ipynb
  └── run.py
```

All scripts are placed under the **scripts** folder, and you can find the code that generates our prediction file pred.csv in `run.py`.


## Implementation Details

#### `'run.py'`
Script that contains the best algorithm implemented, with the generation of corresponding prediction file under `./scripts/data/pred.csv`. It executes the following steps to get the test set prediction:

* Load the training dataset into feature matrix(X), class labels(Y, -1 or 1), and event ids
* Data preprocessing
     
        - Split the data into 4 sub-datasets based on 'PRI_jet_num' (0, 1, 2, 3 for each sub-dataset)
        - impute the missing values with k-means clustering 
        - Standardize the sub-datasets by column means and standard deviations

* Train a ridge regression model with 10-fold cross validation, with an automatic process of finding the best number of clusters, degree, and lambda
* Train a ridge regression model on the complete training set, using the best number of clusters, degree, and lambda
* Load the test dasaset into feature matrix(X) and event ids(ID for reordering prediction)
* Compute and generate a prediction csv file `./scripts/data/pred.csv`


#### `'costs.py'`
Script that contains the functions to calculate the loss functions for implemented machine learning algorithms.


#### `'implementations.py'`
Script that contains the implementation of machine learning algorithms according to the following table:

| Function            | Parameters | Details |
|-------------------- |-----------|---------|
| `least_squares_GD`  | `y, tx, initial_w, max_iters, gamma`  | Linear Regression by Gradient Descent |
| `least_squares_SGD` | `y, tx, initial_w, max_iters, gamma`  | Linear Regression by Stochastic Gradient Descent |
| `least_squares`     | `y, tx` | Linear Regression by Solving Normal Equation |
| `ridge_regression`  | `y, tx, lambda_` | Ridge Regression by Soving Normal Equation |
| `logistic_regression`| `y, tx, initial_w, max_iters, gamma, threshold, batch_size` | Logistic Regression by Stochastic Gradient Descent |
| `reg_logistic_regression` | `y, tx, lambda_, initial_w, max_iters, gamma, threshold, batch_size` | Regularized Logistic Regression by Stochastic Gradient Descent |

All functions returns a set of two key values: `w, loss`, where `w` indicates the last weight vector of the algorithm, and `loss` corresponds to this weight `w`.


#### `'preprocess.py'`
Script that contains functions for preprocessing both the training and the test dataset. 


#### `'proj1_helpers.py'`
Script that contains functins for loading the dataset and creating the prediction files. 


#### `'helpers.py'`
Script that contains functions for basic key value (gradient, hessian, sigmoid, `(w, loss)`) calculation and a batch iteration function for implementing stochastic gradient descent.


#### `'cross_validation.py'`
Script that contains functions to carry-out a k-fold cross validation on the training dataset. 


#### `'project1.ipynb'`
Notebook file contains code demonstrating our training and validation process of implemented machine learning algorithms (stated in `implementation.py`). 
We reserved the block outputs to show the hyperparameter setting with metrics for your reference.


## Best Performance
Our best model: Ridge Regression with imputation through k-means clustering, test accuracy: 0.803, F1 score: 0.685


## Authors
* *Guanqun Liu*
* [*Yixuan Xu*](https://github.com/Alvorecer721)
* [*Xianjie Dai*](https://github.com/xianjiedai)
