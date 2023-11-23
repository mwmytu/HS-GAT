import time
import math
import torch
import random
import datetime
from torch import nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import config
from model_yelp_cpu_test2 import NN
from utils import accuracy, accuracy2, recall, recall2, f1, f12, precision, precision2, delete_stopwords, \
    delete_stopwords2, cut_words_key, get_n, recall_topn, precision_topn, f1_score_topn, \
    get_glove, charToEmbed, roc, precision_topn, recall_topn, hits, recall_topn, get_ndcg, mrr
from utils import drawer
from dataloader import getData, train_deal, test_deal

dataset, train_data, test_data, ug_u_u2, ug_v_v2 = getData()
u_v_matrix, v_u_matrix, target, words_list, u_u_dict, v_v_dict, u_u_dict_all = train_deal(dataset, train_data)

# 对target使用计算权重进行处理
# class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(target), y=target)
# class_weights = torch.tensor(class_weights, dtype=torch.float)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))

_words = delete_stopwords(words_list)
# words截断  保证每个语句长度都相同
_words = cut_words_key(_words, config.cut_len)
# 打开glove文件
dict_words = get_glove(config.glove_path)
# 将每个单词转化为embed
word_embed = charToEmbed(_words, dict_words)
word_embed = torch.Tensor(np.array(word_embed, dtype=np.float))

epochs = config.epochs
# week = len(ratings['week'].unique())
# dis = len(ratings['dis'].unique())

in_channels = config.in_channels
out_channels = config.out_channels

w_out = config.w_out

# 训练集中用户个数和事件个数
user_len_train = u_v_matrix.shape[1]
business_len_train = v_u_matrix.shape[1]

num_embed = config.n_emb
user_len = len(dataset["user_id"].unique())
business_len = len(dataset["business_id"].unique())
model = NN(u_v_matrix, v_u_matrix, in_channels, out_channels, user_len, business_len, num_embed, w_out)
# 加载训练权重
# model.load_state_dict(torch.load("./model/2022-10-31_16_06_10_model.pth"))
# model.load_state_dict(torch.load("./model/2022-12-17_12_45_56_model.pth"))
# model.load_state_dict(torch.load("./model/2022-12-17_19_48_13_model.pth"))
model.load_state_dict(torch.load("./model/2023-11-14_21_07_03_model_0.001_20.pth"))


mse = torch.nn.MSELoss()
model.eval()
u_i_dict = test_deal(dataset, test_data)

# with torch.no_grad():
#     for q in range(50):
#         print("-------------")
#         max_hits, max_recall, max_precision, max_f1 = 0, 0, 0, 0
#         # max_mrr = 0
#         max_hits_sample = 0
#         # min_mrr_sample = 0
#         max_hits_list = [0 for i in range(20)]
#         max_recall_list = [0 for i in range(20)]
#         max_pre_list = [0 for i in range(20)]
#         max_f1_list = [0 for i in range(20)]
#         aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#         for i in range(50):
#
#             output = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_i_dict, v_v_dict)
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
#             aver_recall_list.append(recall_list)
#             aver_pre_list.append(precision_list)
#             aver_f1_list.append(f1_list)
#
#             for j in range(20):
#                 if max_recall_list[j] < recall_list[j]:
#                     max_recall_list[j] = recall_list[j]
#                 if max_pre_list[j] < precision_list[j]:
#                     max_pre_list[j] = precision_list[j]
#                 if max_f1_list[j] < f1_list[j]:
#                     max_f1_list[j] = f1_list[j]
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
#         print("aver recall:{}".format(aver_recall_list_))
#         print("aver precision:{}".format(aver_pre_list_))
#         print("aver f1-score:{}".format(aver_f1_list_))

with torch.no_grad():
    for q in range(100):
        print("-------------")
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):
            output, user_embed, item_embed = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_i_dict, v_v_dict)
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
                hits_, hits_dict = hits(u_i_dict, test_topn_arr2d, N)
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
