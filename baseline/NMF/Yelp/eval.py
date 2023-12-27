import time
import math
import torch
import random
import datetime
from torch import nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import config
from config import n_emb
from model import NMF
from utils import *
from dataloader import getData, train_deal, test_deal


dataset, train_data, test_data = getData()
user_id, item_id, target, u_u_dict = train_deal(dataset, train_data)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
user_id = torch.tensor(user_id)
item_id = torch.tensor(item_id)
num_users = len(user_id)
num_items = len(item_id)

model = NMF(user_id, item_id, n_emb)
model.load_state_dict(torch.load("./model/2022-12-29_14_12_22_model.pth"))

mse = torch.nn.MSELoss()
model.eval()
u_i_dict = test_deal(dataset, test_data)

# with torch.no_grad():
#     for q in range(50):
#         print("-------------")
#         aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#         for i in range(50):
#
#             output = model(user_id, item_id, u_i_dict)
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
    # for q in range(100):
    #     print("-------------")
    #     aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
    for i in range(10):
        output, x_u_ts, x_v_ts = model(user_id, item_id, u_i_dict)
        mse_loss = mse(output, target)
        rmse_loss = math.sqrt(mse_loss)
        output_ = output.detach().numpy()
        # 根据留一法获得测试结果
        test_topn_list2d = []
        # tsne_list2d = []
        for k in u_i_dict.keys():
            topn_list1d = []
            # tsne_list1d = []
            for j in u_i_dict[k]:
                topn_list1d.append(output_[k][j])
                # tsne_list1d.append(x_v_ts[j])
            test_topn_list2d.append(topn_list1d)
            # tsne_list2d.append(tsne_list1d)
        test_topn_arr2d = np.array(test_topn_list2d)
        index = 1
        # test_point_list2d = []
        # test_tsne_list2d = []
        # for i in range(len(test_topn_list2d[0])):
        #     dis = math.sqrt((x_u_ts[0][0] - tsne_list2d[0][i][0]) ** 2 + (x_u_ts[0][1] - tsne_list2d[0][i][1]) ** 2)
        #     test_point_list2d.append([index, test_topn_list2d[0][i], str(u_i_dict[0][i])])
        #     test_tsne_list2d.append([tsne_list2d[0][i][0], tsne_list2d[0][i][1], str(u_i_dict[0][i]) + ',' + str(round(dis, 2))])
        #     index += 1
        N = 1
        hits_list = []
        # ndcg_aver_list = []
        # mrr_list = []
        recall_list = []
        precision_list = []
        f1_list = []

        j = 5
        # for j in range(10, 25, 5):
        hits_, hits_dict = hits(u_i_dict, test_topn_arr2d, j)
        hits_list.append(hits_)
        # 获得用户-item字典的keys
        user_item_keys_list = list(u_i_dict.keys())
        # 查找命中的第一个用户
        hits_user_index = "".join([str(i) for i in hits_]).find("1")
        # 获得当前用户的embed
        hits_user_embed = x_u_ts[user_item_keys_list[hits_user_index]]
        # concat用户与item
        # hits_user_item_embed = torch.concat((x_v_ts, torch.reshape(hits_user_embed, (1, hits_user_embed.shape[0]))), dim=0)
        # 根据命中用户打印候选集
        hits_user_item_list = u_i_dict[user_item_keys_list[hits_user_index]]
        # top-k的item
        top_k_item_list = hits_dict[user_item_keys_list[hits_user_index]]
        # 获得命中用户的poi邻居
        hits_user_item_neibor = u_u_dict[user_item_keys_list[hits_user_index]]
        hits_user_user_neibor = []

    # 获得top-k poi的得分
        top_k_item_score_list = []
        for i in range(len(top_k_item_list)):
            top_k_item_score_list.append(round(output_[user_item_keys_list[hits_user_index]][top_k_item_list[i]], 4))
        print("=================")
        print(hits_user_item_list)
        print(top_k_item_list)
        print(top_k_item_score_list)
        print("=================")
        # 获取用户embed与候选集每个item的embed
        hits_user_item_list2d = []
        hits_user_item_score = []
        for k in range(len(hits_user_item_list)):
            hits_user_item_list2d.append(x_v_ts[hits_user_item_list[k]].detach().numpy().tolist())
            hits_user_item_score.append(round(output_[user_item_keys_list[hits_user_index]][hits_user_item_list[k]], 4))
        # # 命中用户对候选poi的得分
        #
        # hits_user_item_list2d.append(hits_user_embed.detach().numpy().tolist())
        # # 降维后的item嵌入
        hits_tsne_list2d = tsne(np.array(hits_user_item_list2d), 2, 150).tolist()
        # # # 降维后进行聚类
        # hits_cluster_list1d = cluster(np.array(hits_user_item_list2d), int(100 / j)).tolist()
        hits_cluster_list1d = cluster(np.array(hits_tsne_list2d), int(100 / j)).tolist()
        # # 组成matplotlib的数据
        mat_X, mat_Y = [], []
        for k in range(len(hits_tsne_list2d)):
            mat_X.append(hits_tsne_list2d[k][0])
            mat_Y.append(hits_tsne_list2d[k][1])
        # # hits_user_item_list.append(-1)
        # print("====================")
        # print(len(mat_X), len(mat_Y), len(hits_cluster_list1d), len(hits_user_item_list))
        # print("====================")
        drawer("KMeans", "X", "Y", mat_X, mat_Y, "POI", hits_cluster_list1d, hits_user_item_score, True)

# with torch.no_grad():
#     for q in range(50):
#         print("-------------")
#         aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#         for i in range(50):
#             output = model(user_id, item_id, u_i_dict)
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
