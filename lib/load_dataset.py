import os
import numpy as np
import pandas as pd


def load_st_dataset(dataset):
    # output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  # [26208,358,1] ---->[26208,358],按照第一个维度整合
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # [16992, 307, 3]---->[16992, 307]只要流量数据
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # [28224,883,1]---->[28224,883]
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # [17856, 170, 3]---->[17856, 170]
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        data = np.array(pd.read_csv(data_path, header=None))  # [12671, 228]
    elif dataset == 'METR-LA':
        data_path = os.path.join('./data/METR-LA/METR.h5')
        data = pd.read_hdf(data_path)
    elif dataset == 'BJ':
        data_path = os.path.join('./data/BJ/BJ500.csv')
        data = np.array(pd.read_csv(data_path, header=0, index_col=0))
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)  # X=[B,N,D]
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
