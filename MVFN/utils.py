import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset


class StandardScaler(object):
    def __init__(self):
        pass

    def transform(self, mean, std, X):
        X = 1. * (X - mean) / std
        return X

    def inverse_transform(self, mean, std, X):
        X = X * std + mean
        return X

class MinMaxScaler(object):
    def __init__(self):
        pass

    def transform(self, min_val, max_val, X):
        if max_val - min_val == 0:
            return np.zeros_like(X)
        X = (X - min_val) / (max_val - min_val)
        return X

    def inverse_transform(self, min_val, max_val, X):
        X = X * (max_val - min_val) + min_val
        return X    




def read_data(args):
    pick = pd.read_csv(args.pick, header=None, index_col=None).values
    drop = pd.read_csv(args.drop, header=None, index_col=None).values
    data = np.dstack((pick,drop))
    Nodes = len(data[0])

    # val_num, test_num = args.val_rate, args.test_rate
    # total = data.shape[0]
    # train_num = total - val_num - test_num
    # train = data[:train_num, :, :]
    # val = data[train_num:train_num + val_num, :, :]
    # test = data[train_num + val_num:, :, :]

    # train_rate, val_rate = args.train_rate, args.val_rate
    # train, val, test = data[0:-train_rate, :, :], data[-train_rate:-val_rate, :, :], data[-val_rate:, :, :]
    # Nodes = len(data[0])
    
    Nodes = len(data[0])

    return data, Nodes

def graph(args):
    adj_data = pd.read_csv(args.adj_data, header=None, index_col=None).values

    graph_data = torch.FloatTensor(adj_data)
    N = len(graph_data)
    matrix_i = torch.eye(N, dtype=torch.float)  # 定义[N, N]的单位矩阵
    graph_data += matrix_i  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

    return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


def get_data(data, input_dim, output_dim):
    X, Y = [], []
    L = len(data)
    for i in range(L - input_dim - output_dim + 1):
        X.append(data[i:i+input_dim, :, :])
        Y.append(data[i+input_dim:i+input_dim+output_dim, :, :])
    X = np.array(X)
    Y = np.array(Y)
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    return X, Y

def data_process(args, data):
    X, Y = get_data(data, args.input_dim, args.output_dim)
    total = X.shape[0]
    val_num, test_num = int(args.val_rate), int(args.train_rate)
    train_num = total - val_num - test_num

    train_X, train_Y = X[:train_num], Y[:train_num]
    val_X, val_Y = X[train_num:train_num+val_num], Y[train_num:train_num+val_num]
    test_X, test_Y = X[train_num+val_num:], Y[train_num+val_num:]

    print(f"train_X: {train_X.shape}, train_Y: {train_Y.shape}")
    print(f"val_X: {val_X.shape}, val_Y: {val_Y.shape}")
    print(f"test_X: {test_X.shape}, test_Y: {test_Y.shape}")

    train = TensorDataset(train_X, train_Y)
    val = TensorDataset(val_X, val_Y)
    test = TensorDataset(test_X, test_Y)

    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=None, num_workers=0)
    val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=None, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=None, num_workers=0)

    return train_loader, val_loader, test_loader

# 统计参数量（M）
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

# 用于计算平均值
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt