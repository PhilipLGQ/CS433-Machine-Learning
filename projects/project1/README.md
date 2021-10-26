# Machine Learning Project 1 (2021): The Higgs Boson Machine Learning Challenge

The "Higgs Boson" is an elementary particle belonging to the Standard Model of particle physics. It is produced by exerting the quantum excitation of the Higgs field. Due to its rapidness of decay, scientists found it difficult to directly observe its decay signal due to background noise. The goal of this machine learning challenge is to improve the discovery significance of the particle by classifying provided data events into "tau tau decay of Higgs Boson" or "background". The aim of this project is to realize a satisfying performance on binary classification.

Detailed project outlines and requirements can be found in the [project description](./projects/project1/project1_description.pdf). This challenge includes an [AIcrowd online contest] (https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), the original kaggle contest is [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson) (2014).

## Getting Started
Provided scripts and notebook files were built and tested with a conda environment with python version 3.7.11. 
The following external libraries are used within the scripts:

numpy (as np)
matplotlib.pyplot (as plt)


## Running Prerequisites
Before running the scripts and notebook files, you should keep the folder structure under folder **scripts** as follows:

.
├── costs.py
├── cross_validation.py
├── data
│   ├── test.csv               # training set, extract first
│   └── train.csv              # test set, extract first
├── dataplot.py
├── helpers.py
├── implementations.py
├── preprocess.py
├── proj1_helpers.py
├── project1.ipynb
└── run.py

All scripts are placed under the **scripts** folder, and you can find the code that generates our prediction file pred.csv in 'run.py'.


## Script Details

#### 'costs.py'
Script that contains the functions to calculate the loss functions for implemented machine learning algorithms.

#### 'implementations.py'
