import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch import Tensor
from array import array

from config import out_dim, out_dim2, dense_hidden1, dense_hidden2, dense_out


# 用户GAT
class UserGAT(nn.Module):
    def __init__(self):
        super(UserGAT, self).__init__()
        self.gat1 = GATConv(out_dim, out_dim)
        self.gat2 = GATConv(out_dim, out_dim2)

    def forward(self, E, A):
        x1 = F.relu(self.gat1(E, A))
        x1 = F.dropout(x1, p=0.3)
        x2 = F.relu(self.gat2(x1, A))
        return x2


# 项目GAT
class ItemGAT(nn.Module):
    def __init__(self):
        super(ItemGAT, self).__init__()
        self.gat1 = GATConv(out_dim, out_dim)
        self.gat2 = GATConv(out_dim, out_dim2)

    def forward(self, E, A):
        x1 = F.relu(self.gat1(E, A))
        x1 = F.dropout(x1, p=0.3)
        x2 = F.relu(self.gat2(x1, A))
        return x2


class UserLinear(nn.Module):
    def __init__(self, R_U: Tensor):
        super(UserLinear, self).__init__()
        self.linear1 = nn.Linear(out_dim*R_U.shape[1], dense_hidden1)
        self.linear2 = nn.Linear(dense_hidden1, dense_hidden2)
        self.out = nn.Linear(dense_hidden2, dense_out)

    def forward(self, user_embed):
        user_embed = F.relu(self.linear1(user_embed))
        user_embed = F.dropout(user_embed)
        user_embed = F.relu(self.linear2(user_embed))
        user_embed = F.dropout(user_embed)
        user_embed = self.out(user_embed)
        return user_embed


class ItemLinear(nn.Module):
    def __init__(self, R_I: Tensor):
        super(ItemLinear, self).__init__()
        self.linear1 = nn.Linear(out_dim*R_I.shape[1], dense_hidden1)
        self.linear2 = nn.Linear(dense_hidden1, dense_hidden2)
        self.out = nn.Linear(dense_hidden2, dense_out)

    def forward(self, item_embed):
        item_embed = F.relu(self.linear1(item_embed))
        item_embed = F.dropout(item_embed)
        item_embed = F.relu(self.linear2(item_embed))
        item_embed = F.dropout(item_embed)
        item_embed = self.out(item_embed)
        return item_embed


class AGCN(nn.Module):
    def __init__(self, R_U: Tensor, R_V: Tensor):
        super(AGCN, self).__init__()

        # 通过评分矩阵初始化embed
        self.score_embed = nn.Embedding(int(R_U.max())+1, out_dim)
        # 将初始化embed进行降维
        self.user_linear = UserLinear(R_U)
        self.item_linear = ItemLinear(R_V)
        # 用户GCN
        self.user_gcn = UserGAT()
        # 项目GCN
        self.item_gcn = ItemGAT()

    def forward(self, R_U, R_V, A_U, A_V):
        # 初始化用户和项目embed
        user_embed = self.score_embed(R_U)
        item_embed = self.score_embed(R_V)
        user_embed = torch.reshape(user_embed, (R_U.shape[0], R_U.shape[1]*out_dim))
        item_embed = torch.reshape(item_embed, (R_V.shape[0], R_V.shape[1]*out_dim))
        user_embed = self.user_linear(user_embed)
        item_embed = self.item_linear(item_embed)
        # 通过GAT活的最终的embed
        user_embed = self.user_gcn(user_embed, torch.tensor(A_U).long())
        item_embed = self.item_gcn(item_embed, torch.tensor(A_V).long())

        # 得分
        score = torch.matmul(user_embed, item_embed.T)

        return score
