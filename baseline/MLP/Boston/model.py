import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import math
import config
from sklearn.preprocessing import normalize, MinMaxScaler
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):

    def __init__(self, user_id, item_id, num_embed):
        super(MLP, self).__init__()

        self.user_embedding = nn.Embedding(user_id.max()+1, num_embed)
        self.item_embedding = nn.Embedding(item_id.max()+1, num_embed)

        self.dense1 = nn.Linear(num_embed * 2, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 16)
        self.dense4 = nn.Linear(16, config.n_out)
        # PREDICT
        self.out = nn.Linear(config.n_out, config.dense_out)
        # self.dense1 = nn.Linear(config.n_out * 2, config.dense_hidden1)
        # self.dense2 = nn.Linear(config.dense_hidden1, config.dense_hidden2)
        # self.out = nn.Linear(config.dense_hidden2, config.dense_out)

    def forward(self, user_id, item_id, u_u_dict):

        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)

        # 将用户embed和事件embed进行拼接
        u_v_embed_cat = []
        #
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat.append(torch.cat((user_embed[v], item_embed[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat = torch.tensor(u_v_embed_cat)
        for i in range(user_embed.shape[0]):
            for j in range(item_embed.shape[0]):
                u_v_embed_cat.append(torch.cat((user_embed[i], item_embed[j])).tolist())
        u_v_embed_cat = torch.tensor(u_v_embed_cat)

        x = F.dropout(F.relu(self.dense1(u_v_embed_cat)))
        x = F.dropout(F.relu(self.dense2(x)))
        x = F.dropout(F.relu(self.dense3(x)))
        x = F.dropout(F.relu(self.dense4(x)))
        score = F.relu(self.out(x))

        score = torch.reshape(score, (user_embed.shape[0], item_embed.shape[0]))

        return score
