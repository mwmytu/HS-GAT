import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import time

from dataloader import get_data, train_deal, test_deal
from bert import BERTEncoder, get_tokens_and_segments
from model import DeepCLFM
from config import out_dim


# 获得数据集
dataset, train_data, test_data = get_data()
target, u_u_dict, train_data = train_deal(dataset, train_data)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
user_num = len(dataset["user_id"].unique())
item_num = len(dataset["business_id"].unique())
# 获得用户-项目评论集
user_words_df = pd.read_csv("./word_data/user_words_str_30000.csv", encoding="utf8")["words"]
item_words_df = pd.read_csv("./word_data/item_words_str_30000.csv", encoding="utf8")["words"]

user_words_list2d = []
for i in range(user_words_df.shape[0]):
    user_words_list2d.append([int(j) for j in user_words_df.loc[i].split(",")])
user_words_list1d = np.squeeze(np.reshape(np.array(user_words_list2d), (-1, 1))).tolist()
item_words_list2d = []
for i in range(item_words_df.shape[0]):
    item_words_list2d.append([int(j) for j in item_words_df.loc[i].split(",")])
item_words_list1d = np.squeeze(np.reshape(np.array(item_words_list2d), (-1, 1))).tolist()

# BERT预训练得到的embed
user_words_bert_df = pd.read_csv("./word_data/user_bert_text_embed_25.csv", encoding="utf8")["words"]
item_words_bert_df = pd.read_csv("./word_data/item_bert_text_embed_25.csv", encoding="utf8")["words"]
user_words_bert_list2d = []
for i in range(user_words_bert_df.shape[0]):
    user_words_bert_list2d.append([float(j) for j in user_words_bert_df.loc[i].split(",")])
user_words_bert_tensor3d = torch.reshape(torch.tensor(user_words_bert_list2d), (user_num, 100, out_dim))
item_words_bert_list2d = []
for i in range(item_words_bert_df.shape[0]):
    item_words_bert_list2d.append([float(j) for j in item_words_bert_df.loc[i].split(",")])
item_words_bert_tensor3d = torch.reshape(torch.tensor(item_words_bert_list2d), (item_num, 100, out_dim))

# 矩阵分解预训练得到的embed
user_matrix_embed_df = pd.read_csv("./word_data/user_embed_25_30000.csv", encoding="utf8")["user_embed"]
item_matrix_embed_df = pd.read_csv("./word_data/item_embed_25_30000.csv", encoding="utf8")["item_embed"]
user_matrix_list2d = []
for i in range(user_matrix_embed_df.shape[0]):
    user_matrix_list2d.append([float(j) for j in user_matrix_embed_df.loc[i].split(",")])
# user_words_list1d = np.squeeze(np.reshape(np.array(user_words_list2d), (-1, 1))).tolist()
item_matrix_list2d = []
for i in range(item_matrix_embed_df.shape[0]):
    item_matrix_list2d.append([float(j) for j in item_matrix_embed_df.loc[i].split(",")])
LFM_U = torch.tensor(user_matrix_list2d)
LFM_I = torch.tensor(item_matrix_list2d)

# 获得DeepCLFM模型
model = DeepCLFM(user_num, item_num)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mse = nn.MSELoss()
for i in range(50):
    optimizer.zero_grad()
    score = model(user_words_bert_tensor3d, item_words_bert_tensor3d, LFM_U, LFM_I, u_u_dict)
    loss = mse(target, score)
    print("======第{}次迭代======".format(i + 1))
    print("MSE Loss:{0:5}".format(loss.item()))
    # print("RMSE Loss:{0:5}".format(rmse_loss))
    print("======迭代结束======")
    loss.backward()
    optimizer.step()

nowDate = time.strftime("%Y-%m-%d_%H_%M_%S")
torch.save(model.state_dict(), "./model/DeepCLFM-" + nowDate + ".pth")

