import numpy as np


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    """
    :param data: shape [B, N,D]
    :param window:每个样本的输入序列长度，论文中的P
    :param horizon:预测时域长度，即生成的标签序列的长度,horizon = 12,预测接下来12个时间片数据
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)#返回B
    # 确定切割的终止索引
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


#应该是用不到
if __name__ == '__main__':
    from data.load_raw_data import Load_Sydney_Demand_Data
    path = '../data/1h_data_new3.csv'
    data = Load_Sydney_Demand_Data(path)
    print(data.shape)
    X, Y = Add_Window_Horizon(data, horizon=2)
    print(X.shape, Y.shape)


