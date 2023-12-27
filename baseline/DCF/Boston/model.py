import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np

from config import n_out, n_hidden1, n_hidden2


# 用户MLP
class UserMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 用户MLP
        self.linear1 = nn.Linear(self.input_dim, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, self.output_dim)

    def forward(self, input_matrix):
        x1 = F.relu(self.linear1(input_matrix))
        x1 = F.dropout(x1, p=0.3)
        x2 = F.relu(self.linear2(x1))
        x2 = F.dropout(x2, p=0.3)
        out = F.relu(self.out(x2))
        return out


# 项目MLP
class ItemMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ItemMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 项目MLP
        self.linear1 = nn.Linear(self.input_dim, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, self.output_dim)

    def forward(self, input_matrix):
        x1 = F.relu(self.linear1(input_matrix))
        x1 = F.dropout(x1, p=0.3)
        x2 = F.relu(self.linear2(x1))
        x2 = F.dropout(x2, p=0.3)
        out = F.relu(self.out(x2))
        return out


class DCF(nn.Module):
    def __init__(self, user_matrix: Tensor, item_matrix: Tensor):
        super(DCF, self).__init__()
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        # 用户MLP
        self.user_mlp = UserMLP(self.user_matrix.shape[1], n_out)
        # 项目MLP
        self.item_mlp = ItemMLP(self.item_matrix.shape[1], n_out)

    def forward(self, user_matrix, item_matrix, u_u_dict: dict, l):
        # 用户embed
        user_embed = self.user_mlp(user_matrix)
        # 项目embed
        item_embed = self.item_mlp(item_matrix)

        # score = torch.zeros(l, dtype=torch.float32)
        # i = 0
        # for k in u_u_dict.keys():
        #     for j in u_u_dict[k]:
        #         score[i] = torch.dot(user_embed[k], item_embed[j])
        #         i += 1
        score = torch.matmul(user_embed, item_embed.T)
        return score

