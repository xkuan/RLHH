import operator
import random
from functools import reduce
from itertools import permutations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import PDNConv, GATv2Conv, GCNConv
from torch_geometric.nn.pool.glob import global_mean_pool, global_max_pool
# from params import run_mode


class MLP(nn.Module):
    def __init__(self, xdim, ydim, hiddenSize=32):
        super().__init__()
        layer_sizes = [xdim, hiddenSize, hiddenSize]
        self.lin1 = nn.Linear(xdim, hiddenSize)
        self.lin2 = nn.Linear(hiddenSize, xdim)
        self.lin3 = nn.Linear(xdim, ydim)
        layers = reduce(operator.add,
                        [[nn.Linear(a, b),
                          nn.BatchNorm1d(b),
                          nn.ReLU(),
                          nn.Dropout(p=0.2)]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], ydim)]

        # for layer in layers:
        #     if type(layer) == nn.Linear:
        #         nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        # h = x
        # x = self.lin1(x)
        # x = F.relu(x)
        # x = self.lin2(x)
        # x = F.relu(x)
        # x += h
        # x = self.lin3(x)
        # if run_mode == 'debug':
        #     print(x.max(1)[1])
        # x = F.softmax(x, dim=1)
        return x


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/pdn_conv.html
# Pathfinder Discovery Networks for Neural Message Passing
# gat_conv, gatv2_conv
class GNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int=32,
                 add_self_loops: bool = True,
                 drop_edge: float = 0.1,
                 # edge_dim: int,
                 ):
        super(GNN, self).__init__()
        self.drop_edge = drop_edge
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels,
                            add_self_loops=add_self_loops)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels,
                            add_self_loops=add_self_loops)
        self.fc = nn.Linear(in_channels, out_channels)

    def normalize_weight(self, weight):
        max_weight = weight.max()
        min_weight = weight.min()
        norm_weight = (max_weight - weight) / (max_weight - min_weight)
        return norm_weight

    def drop_edge_function(self, edge_index, edge_weight):
        num_keep = int((1-self.drop_edge) * edge_index.shape[1])
        temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
        random.shuffle(temp)
        return edge_index[:, temp], edge_weight[temp]

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr[:, 1]
        edge_weight = self.normalize_weight(edge_weight)
        h = x
        new_index, new_weight= self.drop_edge_function(edge_index, edge_weight)
        x = self.conv1(x, new_index, new_weight)
        # x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x += h
        x = global_max_pool(x, batch=data.batch)
        # if run_mode == 'debug':
        #     print(x.max(1)[1])
        # x = F.softmax(x, dim=1)

        return x

class PDN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 hidden_channels: int=64, layers: int=2):
        super(PDN, self).__init__()
        self.layers = layers
        self.leaky_relu = nn.LeakyReLU()
        self.conv = PDNConv(in_channels, out_channels, edge_dim, hidden_channels)
        self.conv1 = PDNConv(in_channels, 2*in_channels, edge_dim, hidden_channels)
        self.conv2 = PDNConv(2*in_channels, in_channels, edge_dim, hidden_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        # if layers == 1:
        #     self.net = nn.Sequential(
        #         PDNConv(in_channels, out_channels, edge_dim, hidden_channels)
        #     )
        # elif layers == 2:
        #     self.net = nn.Sequential(
        #         PDNConv(in_channels, 2*in_channels, edge_dim, hidden_channels),
        #         PDNConv(2*in_channels, out_channels, edge_dim, hidden_channels)
        #     )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = x
        if self.layers == 1:
            x = self.conv(x, edge_index, edge_attr)
        elif self.layers == 2:
            x = self.conv1(x, edge_index, edge_attr)
            x = self.leaky_relu(x)
            x = self.conv2(x, edge_index, edge_attr)
            x += h
        else:
            return None
        x = global_max_pool(x, batch=data.batch)
        x = self.leaky_relu(x)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':

    # test
    test_gnn = GNN(in_channels=10, out_channels=5, edge_dim=6, hidden_channels=32)
    x_data = torch.randn(6, 10)
    edge_list = [[i,j] for i ,j in permutations(range(6), 2) if torch.rand(1) > 0.2]
    edge_data = torch.randn(len(edge_list), 6)
    y_pre = test_gnn(x_data, torch.tensor(edge_list).T, edge_data)
    print(y_pre)

