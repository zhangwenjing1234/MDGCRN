from distutils.command.config import config
import torch.nn as nn
import math
import numpy as np
import torch

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


# class senet_block(nn.Module):
#     def __init__(self, channel=512, ratio=1):
#         super(dct_channel_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, channel // 4, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channel //4, channel, bias=False),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         # b, c, l = x.size() # (B,C,L)
#         # y = self.avg_pool(x) # (B,C,L) -> (B,C,1)
#         # print("y",y.shape)
#         x = x.permute(0,2,1)
#         b, c, l = x.size()
#         y = self.avg_pool(x).view(b, c) # (B,C,L) ->(B,C,1)
#         # print("y",y.shape)
#         # y = self.fc(y).view(b, c, 96)

#         y = self.fc(y).view(b,c,1)
#         # print("y",y.shape)
#         # return x * y
#         return (x*y).permute(0,2,1)
class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )
        # self.dct_norm = nn.LayerNorm([512], eps=1e-6)

        self.dct_norm = nn.LayerNorm([96], eps=1e-6)  # for lstm on length-wise
        # self.dct_norm = nn.LayerNorm([36], eps=1e-6)#for lstm on length-wise on ill with input =36

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L) (32,96,512)
        # y = self.avg_pool(x) # (B,C,L) -> (B,C,1)

        # y = self.avg_pool(x).view(b, c) # (B,C,L) -> (B,C,1)
        # print("y",y.shape
        # y = self.fc(y).view(b, c, 96)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        '''
        for traffic mission:f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic datasets
        '''

        # lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        # print("lr_weight",lr_weight.shape)
        return x * lr_weight  # result


if __name__ == '__main__':
    tensor = torch.rand(8, 7, 96)
    dct_model = dct_channel_block(channel=96)
    result = dct_model.forward(tensor)
    print("result.shape:", result.shape)

# This would give you a weighted sum tensor with shape [B, N] since we are summing along the OUT dimension.


# node_embeddings1 = torch.randn(307, 10)  # 模拟一个具有307个节点，每个节点10维嵌入的张量
#
# # 直接从 node_embeddings1 获取节点数
# node_num = node_embeddings1[0].shape[1]  # 获取第二维的大小，即每个节点的嵌入维度

# 打印结果
#print("节点数量:", node_num)

# # 初始化一个空的张量
# T_i_D_emb = torch.empty(5, 10)
#
# # 定义一个浮点数
# t_i_d_data = [0.2,0.4,0.6,0.8,1.0]
# # 首先将浮点数转换为张量，然后乘以288，并转换其类型为LongTensor
# T_i_D_emb = (torch.tensor(t_i_d_data) * 5).type(torch.LongTensor)

# print(T_i_D_emb)
# source=torch.randn(2,3,4,5)
# print(source.shape)
# print(source)
# t_i_d_data = source[..., 1]
# print(t_i_d_data.shape)
# print(t_i_d_data)
# print('是否使用服务器.')
# print(torch.cuda.is_available())
# print('使用什么服务器.')
# print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


#
# # 创建一个示例的预测值张量
# pred = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
#                      [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],dtype=torch.float)
#
# # 查看张量的形状
# print("原始预测值张量形状:", pred.shape)
#
# # 定义需要计算平均值的维度
# dims = (0, 1, 2)
#
# # 对指定维度进行平均值计算
# pred_mean = pred.mean(dim=dims)
#
# # 输出结果
# print("\n在维度 {} 上的平均值:".format(dims))
# print(pred_mean)
#
#
#
#
# #数据分割部分
# steps_per_day = 3
# data = np.random.rand(5, 8, 1)
#
# print(data)
# time_ind = [i % steps_per_day / steps_per_day for i in range(data.shape[0])]
# time_in_day = np.tile(time_ind, [1, 8, 1]).transpose((2, 1, 0))
# print(time_in_day)
# abc=[time_in_day]
# abc.append(time_in_day)
# print(abc.append(time_in_day))









# #查看npz数据
# data_path = os.path.join('../data/PEMS04/PEMS04.npz')
# dataset='PEMS04'
# data = np.load(data_path)
# print(data.files)
# print(data['data'].shape)
# print(data['data'])
# data=data['data'][:, :, 0]
# if len(data.shape) == 2:
#     data = np.expand_dims(data, axis=-1)  # X=[B,N,D]
# print('Load  Dataset shaped: ', data.shape, data.max(), data.min(), data.mean(), np.median(data))
#
# #查看csv数据
# # data_path = '../data/PEMS07(M)/V_228.csv'
# # data = pd.read_csv(data_path)
# # print("Data shape:", data.shape)
# # print("Data contents:")
# # print(data)


