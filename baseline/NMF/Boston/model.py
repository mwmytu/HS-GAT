import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import math
import config
from sklearn.preprocessing import normalize, MinMaxScaler
import torch.nn.functional as F
import numpy as np


class NMF(nn.Module):

    def __init__(self, user_id, item_id, num_embed):
        super(NMF, self).__init__()

        # MF
        self.user_embedding_mf = nn.Embedding(user_id.max()+1, num_embed)
        self.item_embedding_mf = nn.Embedding(item_id.max()+1, num_embed)
        # MLP
        self.user_embedding_mlp = nn.Embedding(user_id.max()+1, num_embed)
        self.item_embedding_mlp = nn.Embedding(item_id.max()+1, num_embed)

        self.dense_mlp1 = nn.Linear(num_embed*2, 64)
        self.dense_mlp2 = nn.Linear(64, 32)
        # self.dense_mlp3 = nn.Linear(32, 16)
        self.dense_mlp4 = nn.Linear(32, config.mlp_n_emb)
        # PREDICT
        self.out = nn.Linear(num_embed*2+config.mlp_n_emb, config.dense_out)

    def forward(self, user_id, item_id, u_u_dict):

        # MF
        user_embed_mf = self.user_embedding_mf(user_id)
        item_embed_mf = self.item_embedding_mf(item_id)
        # 将用户embed和事件embed进行拼接
        u_v_embed_cat_mf = []
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat_mf.append(torch.cat((user_embed_mf[v], item_embed_mf[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat_mf = torch.tensor(u_v_embed_cat_mf)
        for i in range(user_embed_mf.shape[0]):
            for j in range(item_embed_mf.shape[0]):
                u_v_embed_cat_mf.append(torch.cat((user_embed_mf[i], item_embed_mf[j])).tolist())
        u_v_embed_cat_mf = torch.tensor(u_v_embed_cat_mf)

        # MLP
        user_embed_mlp = self.user_embedding_mlp(user_id)
        item_embed_mlp = self.item_embedding_mlp(item_id)
        u_v_embed_cat_mlp = []
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat_mlp.append(torch.cat((user_embed_mlp[v], item_embed_mlp[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat_mlp = torch.tensor(u_v_embed_cat_mlp)
        for i in range(user_embed_mlp.shape[0]):
            for j in range(item_embed_mlp.shape[0]):
                u_v_embed_cat_mlp.append(torch.cat((user_embed_mlp[i], item_embed_mlp[j])).tolist())
        u_v_embed_cat_mlp = torch.tensor(u_v_embed_cat_mlp)

        user_item_embed_mlp = F.dropout(F.relu(self.dense_mlp1(u_v_embed_cat_mlp)), p=0.3)
        user_item_embed_mlp = F.dropout(F.relu(self.dense_mlp2(user_item_embed_mlp)), p=0.3)
        # user_item_embed_mlp = F.dropout(F.relu(self.dense_mlp3(user_item_embed_mlp)), p=0.3)
        user_item_embed_mlp = F.dropout(F.relu(self.dense_mlp4(user_item_embed_mlp)), p=0.3)

        user_item_embed = torch.concat((u_v_embed_cat_mf, user_item_embed_mlp), dim=-1)

        score = F.relu(self.out(user_item_embed))
        score = torch.reshape(score, (user_embed_mf.shape[0], item_embed_mf.shape[0]))

        return score
