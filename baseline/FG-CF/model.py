import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import math
import config
from config import out_dim
from sklearn.preprocessing import normalize, MinMaxScaler
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
# from test import getData

from config import words_len, sequence_len


# 语义级别的注意力
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, z):
        # 求出每个元路径下每个元路径对应的注意力权重
        w = self.project(z)  # (M, 1)
        # 求出每个元路径的注意力系数
        # 通过softmax可以求出某个权值的重要程度
        beta = torch.squeeze(torch.softmax(w, dim=0))  # (M, 1)
        # beta = beta.expand((z.shape[0],) +beta.shape)  # (N, M, 1)
        # 返回最终的经过语义处理的每个节点的embed
        return beta  # (N, D * K)


class UserGCN(nn.Module):
    def __init__(self, ):
        super(UserGCN, self).__init__()
        self.gcn = GCNConv(config.out_dim, config.gcn_hidden)
        self.gcn2 = GCNConv(config.gcn_hidden, config.gcn_hidden2)
        self.gcn3 = GCNConv(config.gcn_hidden2, config.gcn_out)

    def forward(self, x, A):
        out1 = F.dropout(F.relu(self.gcn(x, A)), p=0.3)
        out2 = F.dropout(F.relu(self.gcn2(out1, A)), p=0.3)
        out3 = F.dropout(F.relu(self.gcn3(out2, A)), p=0.3)
        return out1, out2, out3


class ItemGCN(nn.Module):
    def __init__(self, ):
        super(ItemGCN, self).__init__()
        self.gcn = GCNConv(config.out_dim, config.gcn_hidden)
        self.gcn2 = GCNConv(config.gcn_hidden, config.gcn_hidden2)
        self.gcn3 = GCNConv(config.gcn_hidden2, config.gcn_out)

    def forward(self, x, A):
        out1 = F.dropout(F.relu(self.gcn(x, A)), p=0.3)
        out2 = F.dropout(F.relu(self.gcn2(out1, A)), p=0.3)
        out3 = F.dropout(F.relu(self.gcn3(out2, A)), p=0.3)
        return out1, out2, out3


class FGCF(nn.Module):

    def __init__(self, user_num, item_num, u_v_tensor1d: Tensor, user_tensor1d: Tensor, item_tensor1d: Tensor):
        super(FGCF, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.u_v_tensor1d = u_v_tensor1d
        self.user_tensor1d = user_tensor1d
        self.item_tensor1d = item_tensor1d

        # 获得用户的embed
        self.user_embed = nn.Embedding(user_tensor1d.max()+1, config.out_dim)
        # 获得项目的embed
        self.item_embed = nn.Embedding(item_tensor1d.max()+1, config.out_dim)
        # 获得用户参与项目的embed
        self.user_item_embed = nn.Embedding(u_v_tensor1d.max()+1, config.out_dim)
        # 注意力机制，获得每个用户参与各个项目的影响因子
        self.attention = Attention(config.out_dim)
        # GCN 两种GCN分别获得用户和项目的embed
        self.user_gcn = UserGCN()
        self.item_gcn = ItemGCN()
        # 全连接层 获得评分
        self.linear1 = nn.Linear(2*(config.out_dim+2*config.gcn_hidden+2*config.gcn_hidden2+config.gcn_out), config.dense_hidden1)
        self.linear2 = nn.Linear(config.dense_hidden1, config.dense_hidden2)
        self.out = nn.Linear(config.dense_hidden2, 1)

    # def forward(self, u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, user, item_i, item_j):
    def forward(self, u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_u_dict):
        # 用户|项目 embed
        user_embeddings = self.user_embed(user_tensor1d)
        poi_embeddings = self.item_embed(item_tensor1d)
        # 社交embed
        social_embed = self.user_item_embed(u_v_tensor1d)
        social_embed = torch.reshape(social_embed, (self.user_num, self.item_num, config.out_dim))
        # 获得影响因子
        social_embed2 = torch.zeros((social_embed.shape[0], social_embed.shape[1], social_embed.shape[2]))
        for cur_user in range(social_embed.shape[0]):
            beta = self.attention(social_embed[cur_user])
            for i in range(len(beta)):
                social_embed2[cur_user] = social_embed[cur_user][i]*beta[i]
        # 将用户-项目交互矩阵按照列相加
        social_embed2d = torch.sum(social_embed2, dim=1)
        # 将用户embed和社交embed相加
        social_embed2d = user_embeddings + social_embed2d
        # 通过GCN获得用户的embed
        eu1, eu2, eu3 = self.user_gcn(social_embed2d, torch.tensor(ug_u_u2).long())
        # 通过GCN获得项目的embed
        el1, el2, el3 = self.item_gcn(poi_embeddings, torch.tensor(ug_v_v2).long())
        # 最终的user_embed
        eu1_1 = torch.concat((eu1, social_embed2d), dim=-1)
        eu2_1 = torch.concat((eu2, eu1), dim=-1)
        eu3_1 = torch.concat((eu3, eu2), dim=-1)
        eu = torch.concat((eu1_1, eu2_1, eu3_1), dim=-1)
        # 最终的item_embed
        el1_1 = torch.concat((el1, poi_embeddings), dim=-1)
        el2_1 = torch.concat((el2, el1), dim=-1)
        el3_1 = torch.concat((el3, el2), dim=-1)
        el = torch.concat((el1_1, el2_1, el3_1), dim=-1)
        # BPR
        # user = eu[user]
        # item_i = el[item_i]
        # item_j = el[item_j]
        # prediction_i = (user * item_i).sum(dim=-1)
        # prediction_j = (user * item_j).sum(dim=-1)
        # l2_regulization = 0.01 * (user ** 2 + item_i ** 2 + item_j ** 2).sum(dim=-1)
        # loss2 = -((prediction_i - prediction_j).sigmoid().log().mean())
        # loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()

        # score = torch.matmul(eu, el.T)

        # # 将用户embed和事件embed进行拼接
        # Dense
        u_v_embed_cat = []
        for i in range(eu.shape[0]):
            for j in range(el.shape[0]):
                u_v_embed_cat.append(torch.cat((eu[i], el[j])).tolist())
        u_v_embed_cat = torch.tensor(u_v_embed_cat)
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat.append(torch.cat((eu[v], el[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat = torch.tensor(u_v_embed_cat)
        # u_v_embed_cat = u_v_embed_cat.requires_grad_(True)
        # for i in range(x_u.shape[0]):
        #     for j in range(x_v.shape[0]):
        #         u_v_embed_cat.append(torch.cat((x_u[i], x_v[j])).tolist())
        u_v_embed_cat = torch.tensor(u_v_embed_cat)
        u_v_embed_cat = u_v_embed_cat.requires_grad_(True)

        x = F.dropout(F.relu(self.linear1(u_v_embed_cat)), p=0.3)
        # x = F.dropout(F.leaky_relu(self.dense1(u_v_embed_cat)), p=0.3)
        x = F.dropout(F.relu(self.linear2(x)), p=0.3)
        # x = F.dropout(F.leaky_relu(self.dense2(x)), p=0.3)
        # score = F.softmax(self.out(x))
        # score = F.sigmoid(self.out(x))
        score = F.relu(self.out(x))
        score = torch.reshape(score, (eu.shape[0], el.shape[0]))

        # return prediction_i, prediction_j, loss, loss2
        return score, eu, el
