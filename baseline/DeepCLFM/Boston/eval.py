import time
import math
import pandas as pd
import torch
import random
import datetime
from torch import nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import config
from config import out_dim
from model import DeepCLFM
from utils import accuracy, accuracy2, recall, recall2, f1, f12, precision, precision2, delete_stopwords, \
    delete_stopwords2, cut_words_key, get_n, recall_topn, precision_topn, f1_score_topn, \
    get_glove, charToEmbed, roc, precision_topn, recall_topn, hits, recall_topn, get_ndcg, mrr
from utils import drawer
from dataloader import get_data, train_deal, test_deal
from bert import BERTEncoder, get_tokens_and_segments

# 获得数据集
dataset, train_data, test_data = get_data()
target, u_u_dict, train_data = train_deal(dataset, train_data)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
user_num = len(dataset["user_id"].unique())
item_num = len(dataset["business_id"].unique())
# 获得用户-项目评论集
user_words_df = pd.read_csv("./word_data/user_words_str_40000.csv", encoding="utf8")["words"]
item_words_df = pd.read_csv("./word_data/item_words_str_40000.csv", encoding="utf8")["words"]

user_words_list2d = []
for i in range(user_words_df.shape[0]):
    user_words_list2d.append([int(j) for j in user_words_df.loc[i].split(",")])
user_words_list1d = np.squeeze(np.reshape(np.array(user_words_list2d), (-1, 1))).tolist()
item_words_list2d = []
for i in range(item_words_df.shape[0]):
    item_words_list2d.append([int(j) for j in item_words_df.loc[i].split(",")])
item_words_list1d = np.squeeze(np.reshape(np.array(item_words_list2d), (-1, 1))).tolist()

# BERT预训练得到的embed
user_words_bert_df = pd.read_csv("./word_data/user_bert_text_embed_10.csv", encoding="utf8")["words"]
item_words_bert_df = pd.read_csv("./word_data/item_bert_text_embed_10.csv", encoding="utf8")["words"]
user_words_bert_list2d = []
for i in range(user_words_bert_df.shape[0]):
    user_words_bert_list2d.append([float(j) for j in user_words_bert_df.loc[i].split(",")])
user_words_bert_tensor3d = torch.reshape(torch.tensor(user_words_bert_list2d), (user_num, 100, out_dim))
item_words_bert_list2d = []
for i in range(item_words_bert_df.shape[0]):
    item_words_bert_list2d.append([float(j) for j in item_words_bert_df.loc[i].split(",")])
item_words_bert_tensor3d = torch.reshape(torch.tensor(item_words_bert_list2d), (item_num, 100, out_dim))

# 矩阵分解预训练得到的embed
user_matrix_embed_df = pd.read_csv("./word_data/user_embed_10_35000.csv", encoding="utf8")["user_embed"]
item_matrix_embed_df = pd.read_csv("./word_data/item_embed_10_35000.csv", encoding="utf8")["item_embed"]
user_matrix_list2d = []
for i in range(user_matrix_embed_df.shape[0]):
    user_matrix_list2d.append([float(j) for j in user_matrix_embed_df.loc[i].split(",")])
# user_words_list1d = np.squeeze(np.reshape(np.array(user_words_list2d), (-1, 1))).tolist()
item_matrix_list2d = []
for i in range(item_matrix_embed_df.shape[0]):
    item_matrix_list2d.append([float(j) for j in item_matrix_embed_df.loc[i].split(",")])
LFM_U = torch.tensor(user_matrix_list2d)
LFM_I = torch.tensor(item_matrix_list2d)

mse = torch.nn.MSELoss()
model = DeepCLFM(user_num, item_num)
# 加载训练权重
model.load_state_dict(torch.load("./model/DeepCLFM-2022-12-18_13_38_58.pth"))

model.eval()
u_i_dict = test_deal(dataset, test_data)

# with torch.no_grad():
#     for q in range(50):
#         print("-------------")
#         max_hits, max_recall, max_precision, max_f1 = 0, 0, 0, 0
#         aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#         max_recall_list = [0 for i in range(20)]
#         max_pre_list = [0 for i in range(20)]
#         max_f1_list = [0 for i in range(20)]
#         for i in range(50):
#             output = model(user_words_bert_tensor3d, item_words_bert_tensor3d, LFM_U, LFM_I, u_i_dict)
#             output = torch.squeeze(output)
#             N = 1
#             hits_list = []
#             recall_list = []
#             precision_list = []
#             f1_list = []
#             for j in range(20):
#                 hits_ = hits(u_i_dict, output, N)
#                 hits_list.append(hits_)
#                 recall_list.append(recall_topn(hits_))
#                 precision_list.append(precision_topn(hits_, N))
#                 f1_list.append(f1_score_topn(recall_topn(hits_), precision_topn(hits_, N)))
#
#                 N += 1
#
#             for j in range(20):
#                 if max_recall_list[j] < recall_list[j]:
#                     max_recall_list[j] = recall_list[j]
#                 if max_pre_list[j] < precision_list[j]:
#                     max_pre_list[j] = precision_list[j]
#                 if max_f1_list[j] < f1_list[j]:
#                     max_f1_list[j] = f1_list[j]
#
#             aver_recall_list.append(recall_list)
#             aver_pre_list.append(precision_list)
#             aver_f1_list.append(f1_list)
#
#         aver_recall_list_, aver_pre_list_, aver_f1_list_ = [], [], []
#         for i in range(len(aver_recall_list[0])):
#             recall_sum, pre_sum, f1_sum = 0, 0, 0
#             for j in range(len(aver_recall_list)):
#                 recall_sum += aver_recall_list[j][i]
#                 pre_sum += aver_pre_list[j][i]
#                 f1_sum += aver_f1_list[j][i]
#             aver_recall_list_.append(round(recall_sum / len(aver_recall_list), 5))
#             aver_pre_list_.append(round(pre_sum / len(aver_pre_list), 5))
#             aver_f1_list_.append(round(f1_sum / len(aver_f1_list), 5))
#         print("aver recall:{}".format(aver_recall_list_))
#         print("aver precision:{}".format(aver_pre_list_))
#         print("aver f1-score:{}".format(aver_f1_list_))


with torch.no_grad():
    for q in range(50):
        print("-------------")
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):
            output = model(user_words_bert_tensor3d, item_words_bert_tensor3d, LFM_U, LFM_I, u_i_dict)
            mse_loss = mse(output, target)
            rmse_loss = math.sqrt(mse_loss)
            output_ = output.detach().numpy()
            # 根据留一法获得测试结果
            test_topn_list2d = []
            for k in u_i_dict.keys():
                topn_list1d = []
                for j in u_i_dict[k]:
                    topn_list1d.append(output_[k][j])
                test_topn_list2d.append(topn_list1d)
            test_topn_arr2d = np.array(test_topn_list2d)
            N = 1
            hits_list = []
            # ndcg_aver_list = []
            # mrr_list = []
            recall_list = []
            precision_list = []
            f1_list = []
            for j in range(20):
                hits_ = hits(u_i_dict, test_topn_arr2d, N)
                hits_list.append(hits_)

                # ndcg_ = get_ndcg(hits_dict, u_i_dict)
                # ndcg_aver_list.append(ndcg_)
                # mrr_ = mrr(hits_dict, u_i_dict)
                # mrr_list.append(mrr_)
                recall_list.append(recall_topn(hits_))
                precision_list.append(precision_topn(hits_, N))
                f1_list.append(f1_score_topn(recall_topn(hits_), precision_topn(hits_, N)))

                N += 1

            aver_recall_list.append(recall_list)
            aver_pre_list.append(precision_list)
            aver_f1_list.append(f1_list)

        aver_recall_list_, aver_pre_list_, aver_f1_list_ = [], [], []
        for i in range(len(aver_recall_list[0])):
            recall_sum, pre_sum, f1_sum = 0, 0, 0
            for j in range(len(aver_recall_list)):
                recall_sum += aver_recall_list[j][i]
                pre_sum += aver_pre_list[j][i]
                f1_sum += aver_f1_list[j][i]
            aver_recall_list_.append(round(recall_sum / len(aver_recall_list), 5))
            aver_pre_list_.append(round(pre_sum / len(aver_pre_list), 5))
            aver_f1_list_.append(round(f1_sum / len(aver_f1_list), 5))

        print("aver recall:{}".format(aver_recall_list_))
        print("aver precision:{}".format(aver_pre_list_))
        print("aver f1-score:{}".format(aver_f1_list_))

        print("mse:{}".format(mse_loss))
        print("rmse:{}".format(rmse_loss))

