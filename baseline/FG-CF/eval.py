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
from utils import *

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

model = FGCF(user_num, item_num, u_v_tensor1d, user_tensor1d, item_tensor1d)
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=["acc"])
model.load_state_dict(torch.load("./model/FG-CF-2022-12-28_23_51_57.pth"))
# print(model)


# 获取用户参与过和没有参与过的三元组
# def bpr_data(test_dict, all_rating):
#     features_fill = []
#     for user_id in test_dict.keys():
#         positive_list = [test_dict[user_id][-1]]  # self.train_dict[user_id]
#         all_positive_list = all_rating[user_id]
#         for item_i in positive_list:
#             item_j_list = random.sample(all_positive_list, 99)
#             for item_j in item_j_list:
#                 features_fill.append([user_id, item_i, item_j])
#
#     return features_fill

u_i_dict = test_deal(dataset, test_data)
# user_brp_data2d = bpr_data(u_i_dict, u_u_dict_all)

mse = nn.MSELoss()
model.eval()
# u_i_dict = test_deal(dataset, test_data)

with torch.no_grad():
    for q in range(10):
        output, x_u_ts, x_v_ts = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_u_dict)

        test_topn_list2d = []
        output_ = output.detach().numpy()
        for k in u_i_dict.keys():
            topn_list1d = []
            tsne_list1d = []
            for j in u_i_dict[k]:
                topn_list1d.append(output_[k][j])
                # tsne_list1d.append(x_v_ts[j])
            test_topn_list2d.append(topn_list1d)
            # tsne_list2d.append(tsne_list1d)
        test_topn_arr2d = np.array(test_topn_list2d)
        j = 5
        hits_, hits_dict = hits(u_i_dict, test_topn_arr2d, j)
        user_list = list(hits_dict.keys())
        hits_user_list = []
        for i in range(len(hits_)):
            if hits_[i] == 1:
                hits_user_list.append(user_list[i])
        # 过滤到数组中为0的元素
        user_join_item_list, user_join_item_index_list2d = [], []
        for i in range(len(hits_user_list)):
            user_join_item_list.append(list(filter(lambda x: x != 0, target[hits_user_list[i]].tolist())))
            user_join_item_index_list = []
            for j in range(len(output_[0])):
                if target[hits_user_list[i]][j] != 0:
                    user_join_item_index_list.append(j)
            user_join_item_index_list2d.append(user_join_item_index_list)

        # 预测
        test_user_join_item_list2d = []
        for i in range(len(user_join_item_index_list2d)):
            test_user_join_item_list = []
            for j in range(len(user_join_item_index_list2d[i])):
                test_user_join_item_list.append(output_[hits_user_list[i]][j])
            test_user_join_item_list2d.append(test_user_join_item_list)
        print(output)
        # 过滤掉长度<=2
        new_user_join_item_list2d, new_test_user_join_item_list2d = [], []
        for i in range(len(user_join_item_list)):
            if len(user_join_item_list[i]) >= 3:
                new_user_join_item_list2d.append(user_join_item_list[i])
                new_test_user_join_item_list2d.append(test_user_join_item_list2d[i])
        # 绘制颜色矩阵
        user_join_item_list3d = []
        user_join_item_list3d.append(new_user_join_item_list2d)
        user_join_item_list3d.append(new_test_user_join_item_list2d)
        # 绘制颜色矩阵
        draw_color_matrix(user_join_item_list3d, x_title="POI", y_title="User", x_dis=len(new_user_join_item_list2d),
                          y_dis=len(user_join_item_list3d),
                          save_path="../images/plt/train_user_poi.png",
                          display_ticks=True)

# with torch.no_grad():
#     # for q in range(100):
#     #     print("-------------")
#     #     aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#     for i in range(10):
#         output, x_u_ts, x_v_ts = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_i_dict)
#         mse_loss = mse(output, target)
#         rmse_loss = math.sqrt(mse_loss)
#         output_ = output.detach().numpy()
#         # 根据留一法获得测试结果
#         test_topn_list2d = []
#         # tsne_list2d = []
#         for k in u_i_dict.keys():
#             topn_list1d = []
#             # tsne_list1d = []
#             for j in u_i_dict[k]:
#                 topn_list1d.append(output_[k][j])
#                 # tsne_list1d.append(x_v_ts[j])
#             test_topn_list2d.append(topn_list1d)
#             # tsne_list2d.append(tsne_list1d)
#         test_topn_arr2d = np.array(test_topn_list2d)
#         index = 1
#         # test_point_list2d = []
#         # test_tsne_list2d = []
#         # for i in range(len(test_topn_list2d[0])):
#         #     dis = math.sqrt((x_u_ts[0][0] - tsne_list2d[0][i][0]) ** 2 + (x_u_ts[0][1] - tsne_list2d[0][i][1]) ** 2)
#         #     test_point_list2d.append([index, test_topn_list2d[0][i], str(u_i_dict[0][i])])
#         #     test_tsne_list2d.append([tsne_list2d[0][i][0], tsne_list2d[0][i][1], str(u_i_dict[0][i]) + ',' + str(round(dis, 2))])
#         #     index += 1
#         N = 1
#         hits_list = []
#         # ndcg_aver_list = []
#         # mrr_list = []
#         recall_list = []
#         precision_list = []
#         f1_list = []
#
#         j = 5
#         # for j in range(10, 25, 5):
#         hits_, hits_dict = hits(u_i_dict, test_topn_arr2d, j)
#         hits_list.append(hits_)
#         # 获得用户-item字典的keys
#         user_item_keys_list = list(u_i_dict.keys())
#         # 查找命中的第一个用户
#         hits_user_index = "".join([str(i) for i in hits_]).find("1")
#         # 获得当前用户的embed
#         hits_user_embed = x_u_ts[user_item_keys_list[hits_user_index]]
#         # concat用户与item
#         # hits_user_item_embed = torch.concat((x_v_ts, torch.reshape(hits_user_embed, (1, hits_user_embed.shape[0]))), dim=0)
#         # 根据命中用户打印候选集
#         hits_user_item_list = u_i_dict[user_item_keys_list[hits_user_index]]
#         # top-k的item
#         top_k_item_list = hits_dict[user_item_keys_list[hits_user_index]]
#         # 获得命中用户的poi邻居
#         hits_user_item_neibor = u_u_dict[user_item_keys_list[hits_user_index]]
#         hits_user_user_neibor = []
#
#     # 获得top-k poi的得分
#         top_k_item_score_list = []
#         for i in range(len(top_k_item_list)):
#             top_k_item_score_list.append(round(output_[user_item_keys_list[hits_user_index]][top_k_item_list[i]], 4))
#         print("=================")
#         print(hits_user_item_list)
#         print(top_k_item_list)
#         print(top_k_item_score_list)
#         print("=================")
#         # 获取用户embed与候选集每个item的embed
#         hits_user_item_list2d = []
#         hits_user_item_score = []
#         for k in range(len(hits_user_item_list)):
#             hits_user_item_list2d.append(x_v_ts[hits_user_item_list[k]].detach().numpy().tolist())
#             hits_user_item_score.append(round(output_[user_item_keys_list[hits_user_index]][hits_user_item_list[k]], 4))
#         # # 命中用户对候选poi的得分
#         #
#         # hits_user_item_list2d.append(hits_user_embed.detach().numpy().tolist())
#         # # 降维后的item嵌入
#         hits_tsne_list2d = tsne(np.array(hits_user_item_list2d), 2, 150).tolist()
#         # # # 降维后进行聚类
#         # hits_cluster_list1d = cluster(np.array(hits_user_item_list2d), int(100 / j)).tolist()
#         hits_cluster_list1d = cluster(np.array(hits_tsne_list2d), int(100 / j)).tolist()
#         # # 组成matplotlib的数据
#         mat_X, mat_Y = [], []
#         for k in range(len(hits_tsne_list2d)):
#             mat_X.append(hits_tsne_list2d[k][0])
#             mat_Y.append(hits_tsne_list2d[k][1])
#         # # hits_user_item_list.append(-1)
#         # print("====================")
#         # print(len(mat_X), len(mat_Y), len(hits_cluster_list1d), len(hits_user_item_list))
#         # print("====================")
#         drawer("KMeans", "X", "Y", mat_X, mat_Y, "POI", hits_cluster_list1d, hits_user_item_score, True)

# with torch.no_grad():
#     for q in range(50):
#         print("-------------")
#         aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#         for i in range(50):
#             output = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_i_dict)
#             mse_loss = mse(output, target)
#             rmse_loss = math.sqrt(mse_loss)
#             output_ = output.detach().numpy()
#             # 根据留一法获得测试结果
#             test_topn_list2d = []
#             for k in u_i_dict.keys():
#                 topn_list1d = []
#                 for j in u_i_dict[k]:
#                     topn_list1d.append(output_[k][j])
#                 test_topn_list2d.append(topn_list1d)
#             test_topn_arr2d = np.array(test_topn_list2d)
#             N = 1
#             hits_list = []
#             # ndcg_aver_list = []
#             # mrr_list = []
#             recall_list = []
#             precision_list = []
#             f1_list = []
#             for j in range(20):
#                 hits_ = hits(u_i_dict, test_topn_arr2d, N)
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
#
#         print("mse:{}".format(mse_loss))
#         print("rmse:{}".format(rmse_loss))

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
