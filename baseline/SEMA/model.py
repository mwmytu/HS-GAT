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
# from test import getData


from config import words_len, sequence_len


class SEMA(nn.Module):

    def __init__(self, user_item_word_tensor, item_user_word_tensor, user_dim, item_dim, user_num,
                 item_num, user_latent_factors, item_latent_factors):
        super(SEMA, self).__init__()

        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_num = user_num
        self.item_num = item_num
        self.user_latent_factors = user_latent_factors
        self.item_latent_factors = item_latent_factors

        self.user_embedding = nn.Embedding(user_item_word_tensor.max() + 1, out_dim)
        self.item_embedding = nn.Embedding(item_user_word_tensor.max() + 1, out_dim)

        self.user_id_embedding = nn.Embedding(user_latent_factors[0].max()+1, out_dim)
        # self.user_review_count_embedding = nn.Embedding(user_latent_factors[1].max()+1, out_dim)
        self.item_id_embedding = nn.Embedding(item_latent_factors[0].max()+1, out_dim)
        # self.item_review_count_embedding = nn.Embedding(item_latent_factors[1].max()+1, out_dim)

        # self.user_lstm1 = nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False)
        self.user_lstm1 = [
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False)
        ]
        # self.user_lstm1_1 = nn.LSTM(20, return_sequences=True)
        # self.user_lstm1_2 = nn.LSTM(20)
        self.user_lstm2 = nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False)

        # self.item_lstm1 = nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False)
        self.item_lstm1 = [
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False),
            nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False)
        ]
        # self.item_lstm1_2 = nn.LSTM(20)
        self.item_lstm2 = nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=2, batch_first=False)
        # self.item_lstm2_2 = nn.LSTM(20)

        self.dense1 = nn.Linear(config.n_out * 2, config.dense_hidden1)
        self.dense2 = nn.Linear(config.dense_hidden1, config.dense_hidden2)
        self.out = nn.Linear(config.dense_hidden2, config.dense_out)

    def get_user_words_embed(self, user_embed, i):
        out, (h1, c) = self.user_lstm1[i](user_embed[i])
        return h1[-1]

    def get_item_words_embed(self, item_embed, i):
        out, (h1, c) = self.item_lstm1[i](item_embed[i])
        return h1[-1]

    def forward(self, user_inputs, item_inputs, user_latent_factors, item_latent_factors, u_u_dict):
        # 用户评论文本
        user_embed = self.user_embedding(user_inputs)
        user_embed = torch.reshape(user_embed, (self.user_num, sequence_len, words_len, out_dim))
        # 交换维度
        user_embed = torch.permute(user_embed, (1, 2, 0, 3))

        # user_embed_list = []
        # for i in range(sequence_len):
        #     out, (h1, c) = self.user_lstm1[i](user_embed[i])
        #     words_embed = h1[-1]
        #     user_embed_list.append(words_embed)
        # user_embed_list = torch.tensor([item.detach().numpy() for item in user_embed_list])
        user_words_embed = torch.concat((
            torch.unsqueeze(self.get_user_words_embed(user_embed, 0), dim=0),
            torch.unsqueeze(self.get_user_words_embed(user_embed, 1), dim=0),
            torch.unsqueeze(self.get_user_words_embed(user_embed, 2), dim=0),
            torch.unsqueeze(self.get_user_words_embed(user_embed, 3), dim=0),
            torch.unsqueeze(self.get_user_words_embed(user_embed, 4), dim=0)
        ), dim=0)
        # user_embed_list = torch.permute(user_embed_list, (0, 1, 2))

        # 获得语义级别的embed
        out, (h2, c) = self.user_lstm2(user_words_embed)
        user_embed = h2[-1]
        # 获得用户潜在因素的embed
        user_id_embed = self.user_id_embedding(user_latent_factors[0])
        # user_review_count_embed = self.user_review_count_embedding(user_latent_factors[1])
        # user_embed_ = torch.cat((user_embed, user_id_embed, user_review_count_embed), dim=1)
        user_embed_ = torch.concat((user_embed, user_id_embed), dim=1)
        # 项目评论文本
        item_embed = self.item_embedding(item_inputs)
        item_embed = torch.reshape(item_embed, (self.item_num, sequence_len, words_len, out_dim))
        # 交换维度
        item_embed = torch.permute(item_embed, (1, 2, 0, 3))

        # item_embed_list = []
        # for i in range(sequence_len):
        #     out, (h1, c) = self.item_lstm1[i](item_embed[i])
        #     words_embed = h1[-1]
        #     item_embed_list.append(words_embed)
        # item_embed_list = torch.tensor([item.detach().numpy() for item in item_embed_list])
        item_words_embed = torch.concat((
            torch.unsqueeze(self.get_item_words_embed(item_embed, 0), dim=0),
            torch.unsqueeze(self.get_item_words_embed(item_embed, 1), dim=0),
            torch.unsqueeze(self.get_item_words_embed(item_embed, 2), dim=0),
            torch.unsqueeze(self.get_item_words_embed(item_embed, 3), dim=0),
            torch.unsqueeze(self.get_item_words_embed(item_embed, 4), dim=0)
        ), dim=0)
        # item_embed_list = torch.permute(item_embed_list, (0, 1, 2))

        # 获得语义级别的embed
        out, (h2, c) = self.user_lstm2(item_words_embed)
        item_embed = h2[-1]
        # 获得用户潜在因素的embed
        item_id_embed = self.item_id_embedding(item_latent_factors[0])
        # item_review_count_embed = self.item_review_count_embedding(item_latent_factors[1])
        # item_embed_ = torch.cat((item_embed, item_id_embed, item_review_count_embed), dim=1)
        item_embed_ = torch.cat((item_embed, item_id_embed), dim=1)

        # 将用户embed和事件embed进行拼接
        u_v_embed_cat = []
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat.append(torch.cat((user_embed_[v], item_embed_[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat = torch.tensor(u_v_embed_cat)
        for i in range(user_embed_.shape[0]):
            for j in range(item_embed_.shape[0]):
                u_v_embed_cat.append(torch.cat((user_embed_[i], item_embed_[j])).tolist())
        u_v_embed_cat = torch.tensor(u_v_embed_cat)

        x = F.dropout(F.relu(self.dense1(u_v_embed_cat)), p=0.5)
        # x = F.dropout(F.leaky_relu(self.dense1(u_v_embed_cat)), p=0.3)
        x = F.dropout(F.relu(self.dense2(x)), p=0.5)
        # x = F.dropout(F.leaky_relu(self.dense2(x)), p=0.3)
        # score = F.softmax(self.out(x))
        # score = F.sigmoid(self.out(x))
        score = F.relu(self.out(x))

        score = torch.reshape(score, (user_embed_.shape[0], item_embed_.shape[0]))

        return score
