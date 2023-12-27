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
from model import SEMA
from utils import accuracy, accuracy2, recall, recall2, f1, f12, precision, precision2, delete_stopwords, \
    delete_stopwords2, cut_words_key, get_n, recall_topn, precision_topn, f1_score_topn, \
    get_glove, charToEmbed, roc, precision_topn, recall_topn, hits, recall_topn, get_ndcg, mrr
from utils import drawer
from dataloader import get_data, train_deal, test_deal

dataset, train_data, test_data = get_data()
user_item_word_list, item_user_word_list, target, u_u_dict, train_data, user_latent_factors, item_latent_factors = train_deal(
    dataset, train_data)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
# user_item_word_list_, item_user_word_list_ = [], []
# for i in range(len(user_item_word_list)):
#     for j in [int(j) for j in sum(user_item_word_list[i], [])]:
#         user_item_word_list_.append(j)
# for i in range(len(item_user_word_list)):
#     for j in [int(j) for j in sum(item_user_word_list[i], [])]:
#         item_user_word_list_.append(j)
#
# user_item_words_df = pd.DataFrame(user_item_word_list_, columns=["words"])
# item_user_words_df = pd.DataFrame(item_user_word_list_, columns=["words"])

user_item_words_df = pd.read_csv("./word_data/user_item_words.csv", encoding="utf8")
item_user_words_df = pd.read_csv("./word_data/item_user_words.csv", encoding="utf8")
user_item_word_list_ = user_item_words_df["words"].to_list()
item_user_word_list_ = item_user_words_df["words"].to_list()

user_dim = len(user_item_word_list_)
item_dim = len(item_user_word_list_)
user_num = len(user_item_word_list)
item_num = len(item_user_word_list)

user_item_word_tensor = torch.tensor(user_item_word_list_)
item_user_word_tensor = torch.tensor(item_user_word_list_)
user_latent_factors_tensor = torch.tensor(user_latent_factors)
item_latent_factors_tensor = torch.tensor(item_latent_factors)

# 对target使用计算权重进行处理
# class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(target), y=target)
# class_weights = torch.tensor(class_weights, dtype=torch.float)

epochs = config.epochs
# week = len(ratings['week'].unique())
# dis = len(ratings['dis'].unique())

model = SEMA(user_item_word_tensor, item_user_word_tensor, user_dim, item_dim, user_num, item_num,
             user_latent_factors_tensor, item_latent_factors_tensor)
# 加载训练权重
# model.load_state_dict(torch.load("./model/2022-10-31_16_06_10_model.pth"))
# model.load_state_dict(torch.load("./model/SEMA-2022-11-15_15_31_07.pth"))
model.load_state_dict(torch.load("./model/SEMA-2022-12-28_19_29_54.pth"))

mse = torch.nn.MSELoss()
model.eval()
u_i_dict = test_deal(dataset, test_data)

# with torch.no_grad():
#     for g in range(50):
#         print("-------------")
#         max_hits, max_recall, max_precision, max_f1 = 0, 0, 0, 0
#         aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#         max_recall_list = [0 for i in range(20)]
#         max_pre_list = [0 for i in range(20)]
#         max_f1_list = [0 for i in range(20)]
#         for i in range(50):
#             output = model(user_item_word_tensor, item_user_word_tensor, user_latent_factors_tensor,
#                            item_latent_factors_tensor,
#                            u_i_dict)
#             output = torch.squeeze(output)
#             N = 1
#             hits_list = []
#             # ndcg_aver_list = []
#             # mrr_list = []
#             recall_list = []
#             precision_list = []
#             f1_list = []
#             for j in range(20):
#                 hits_ = hits(u_i_dict, output, N)
#                 hits_list.append(hits_)
#
#                 # ndcg_ = get_ndcg(hits_dict, u_i_dict)
#                 # ndcg_aver_list.append(ndcg_)
#                 # mrr_ = mrr(hits_dict, u_i_dict)
#                 # mrr_list.append(mrr_)
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
#
#         # print("bast recall:{}".format(max_recall_list))
#         # print("bast precision:{}".format(max_pre_list))
#         # print("bast f1-score:{}".format(max_f1_list))
#         print("aver recall:{}".format(aver_recall_list_))
#         print("aver precision:{}".format(aver_pre_list_))
#         print("aver f1-score:{}".format(aver_f1_list_))

with torch.no_grad():
    for q in range(50):
        print("-------------")
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):
            output = model(user_item_word_tensor, item_user_word_tensor, user_latent_factors_tensor,
                           item_latent_factors_tensor, u_i_dict)
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
