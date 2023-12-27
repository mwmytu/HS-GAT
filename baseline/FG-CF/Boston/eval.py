import time
import torch
import datetime
from torch import nn
import numpy as np
import pandas as pd
import math
from sklearn.utils.class_weight import compute_class_weight

from dataloader import get_data, train_deal, test_deal
from model import FGCF
from config import epochs
import data_utils
import random
from utils import hits, recall_topn, precision_topn, f1_score_topn

dataset, train_data, test_data, u_v_arr2d_, ug_u_u2, ug_v_v2 = get_data()
target, u_u_dict, train_data, u_u_dict_all = train_deal(dataset, train_data)

u_i_counters = 0
for i in u_u_dict.keys():
    for j in u_u_dict[i]:
        u_i_counters += u_i_counters

target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
user_num = len(dataset["user_id"].unique())
item_num = len(dataset["business_id"].unique())
u_v_tensor1d = torch.squeeze(torch.tensor(np.reshape(u_v_arr2d_, (1, -1))))
user_tensor1d = torch.tensor(dataset["user_id"].unique().tolist())
item_tensor1d = torch.tensor(dataset["business_id"].unique().tolist())

train_dataset = data_utils.BPRData(
    train_dict=u_u_dict, num_item=item_num, num_ng=5, is_training=True, \
    data_set_count=u_i_counters, all_rating=u_u_dict_all)

po_npo_dataset = train_dataset.ng_sample()

testing_dataset_loss = data_utils.BPRData(
    train_dict=u_u_dict, num_item=item_num, num_ng=5, is_training=True, \
    data_set_count=u_i_counters, all_rating=u_u_dict_all)

mse = torch.nn.MSELoss()
model = FGCF(user_num, item_num, u_v_tensor1d, user_tensor1d, item_tensor1d)
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=["acc"])
model.load_state_dict(torch.load("./model/FG-CF-2023-01-17_15_04_55.pth"))
# print(model)

# 获取用户参与过和没有参与过的三元组
def bpr_data(test_dict, all_rating):
    features_fill = []
    for user_id in test_dict.keys():
        positive_list = [test_dict[user_id][-1]]  # self.train_dict[user_id]
        all_positive_list = all_rating[user_id]
        for item_i in positive_list:
            item_j_list = random.sample(all_positive_list, 99)
            for item_j in item_j_list:
                features_fill.append([user_id, item_i, item_j])

    return features_fill

u_i_dict = test_deal(dataset, test_data)
user_brp_data2d = bpr_data(u_i_dict, u_u_dict_all)

model.eval()
u_i_dict = test_deal(dataset, test_data)

with torch.no_grad():
    for q in range(50):
        print("-------------")
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):
            output = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_i_dict)
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


# with torch.no_grad():
#     for q in range(100):
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
#             # print("第{}次测试".format(i))
#             output = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_i_dict)
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
#     # print("bast mrr:{}".format(max_mrr))
#     #     print("bast recall:{}".format(max_recall_list))
#     #     print("bast precision:{}".format(max_pre_list))
#     #     print("bast f1-score:{}".format(max_f1_list))
#         print("aver recall:{}".format(aver_recall_list_))
#         print("aver precision:{}".format(aver_pre_list_))
#         print("aver f1-score:{}".format(aver_f1_list_))

# print('--------training processing-------')
# with torch.no_grad():
#     max_hits, max_recall, max_precision, max_f1 = 0, 0, 0, 0
#     aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#     max_recall_list = [0 for i in range(20)]
#     max_pre_list = [0 for i in range(20)]
#     max_f1_list = [0 for i in range(20)]
#     for i in range(2):
#         print("-------------")
#         print("第{}次测试".format(i))
#         test_score_list = []
#         k = 0
#         u_i_list, u_i_score_list, u_i_list2d, u_i_score_list2d = [], [], [], []
#         i_score_num = 0
#         for user, item_i, item_j in user_brp_data2d:
#             k += 1
#             prediction_i, prediction_j, loss, loss2 = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2,
#                                                             ug_v_v2,
#                                                             user, item_i, item_j)
#             # i_score_num += prediction_i
#             # u_i_list.append(item_j)
#             u_i_score_list.append(prediction_j)
#             if k % 99 == 0:
#                 # u_i_list2d.append(u_i_list + [item_i])
#                 u_i_score_list2d.append(u_i_score_list + [prediction_i])
#                 u_i_list, u_i_score_list = [], []
#                 # i_score_num = 0
#         output = torch.tensor(u_i_score_list2d)
#         N = 1
#         hits_list = []
#         # ndcg_aver_list = []
#         # mrr_list = []
#         recall_list = []
#         precision_list = []
#         f1_list = []
#         for j in range(20):
#             hits_ = hits(u_i_dict, output, N)
#             hits_list.append(hits_)
#
#             # ndcg_ = get_ndcg(hits_dict, u_i_dict)
#             # ndcg_aver_list.append(ndcg_)
#             # mrr_ = mrr(hits_dict, u_i_dict)
#             # mrr_list.append(mrr_)
#             recall_list.append(recall_topn(hits_))
#             precision_list.append(precision_topn(hits_, N))
#             f1_list.append(f1_score_topn(recall_topn(hits_), precision_topn(hits_, N)))
#
#             N += 1
#
#         for j in range(20):
#             if max_recall_list[j] < recall_list[j]:
#                 max_recall_list[j] = recall_list[j]
#             if max_pre_list[j] < precision_list[j]:
#                 max_pre_list[j] = precision_list[j]
#             if max_f1_list[j] < f1_list[j]:
#                 max_f1_list[j] = f1_list[j]
#
#         aver_recall_list.append(recall_list)
#         aver_pre_list.append(precision_list)
#         aver_f1_list.append(f1_list)
#
#     aver_recall_list_, aver_pre_list_, aver_f1_list_ = [], [], []
#     for i in range(len(aver_recall_list[0])):
#         recall_sum, pre_sum, f1_sum = 0, 0, 0
#         for j in range(len(aver_recall_list)):
#             recall_sum += aver_recall_list[j][i]
#             pre_sum += aver_pre_list[j][i]
#             f1_sum += aver_f1_list[j][i]
#         aver_recall_list_.append(round(recall_sum / len(aver_recall_list), 5))
#         aver_pre_list_.append(round(pre_sum / len(aver_pre_list), 5))
#         aver_f1_list_.append(round(f1_sum / len(aver_f1_list), 5))
#
# print("bast recall:{}".format(max_recall_list))
# print("bast precision:{}".format(max_pre_list))
# print("bast f1-score:{}".format(max_f1_list))
# print("aver recall:{}".format(aver_recall_list_))
# print("aver precision:{}".format(aver_pre_list_))
# print("aver f1-score:{}".format(aver_f1_list_))
