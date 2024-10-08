# Multi-scale Fusion Dynamic Graph Convolutional Recurrent Network for Traffic Forecasting

This is a PyTorch implementation of Multi-scale Fusion Dynamic Graph Convolutional Recurrent Network for Traffic Forecasting, Junbi Xiao, Wenjing Zhang, Wenchao Weng, Yuhao Zhou*, Yunhuan Cong.

## Table of Contents

* configs: training Configs and model configs for each dataset

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our model 

# Data Preparation

The dataset can be downloaded from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).

Unzip the downloaded dataset files to the main file directory, the same directory as run.py.

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
python run.py --datasets {DATASET_NAME} --mode {MODE_NAME}
```
Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD8`

such as `python run.py --datasets PEMSD4`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the 'pre-trained' folder.


