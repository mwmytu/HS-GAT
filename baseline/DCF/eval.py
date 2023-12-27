import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import time
import math

from torch_geometric.graphgym.utils import epoch

from dataloader import getData, train_deal, test_deal, get_matrix_factor
from model import DCF
from config import epochs
from utils import *

dataset, train_data, test_data = getData()
u_v_matrix, v_u_matrix, uv_tag, words_list, u_u_dict, v_v_dict = train_deal(dataset, train_data)
user_item_matrix = get_matrix_factor(dataset, train_data)
user_item_tensor2d = torch.tensor(user_item_matrix, dtype=torch.float32)
target = torch.tensor(user_item_matrix, dtype=torch.float32)
# 获得模型
mse = nn.MSELoss()
model = DCF(user_item_tensor2d, user_item_tensor2d.T)
model.load_state_dict(torch.load("./model/DCF-2022-12-28_00_19_48.pth"))
model.eval()
u_i_dict = test_deal(dataset, test_data)
l = 0
for k in u_i_dict.keys():
    for j in u_i_dict[k]:
        l += 1

with torch.no_grad():
    for q in range(50):
        output, x_u_ts, x_v_ts = model(user_item_tensor2d, user_item_tensor2d.T, u_u_dict, l)

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
#         output, x_u_ts, x_v_ts = model(user_item_tensor2d, user_item_tensor2d.T, u_i_dict, l)
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
#             top_k_item_score_list.append(round(output_[user_item_keys_list[hits_user_index]][top_k_item_list[i]], 5))
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
#             hits_user_item_score.append(round(output_[user_item_keys_list[hits_user_index]][hits_user_item_list[k]], 5))
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
#     aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#     for q in range(50):
#         print("-----------")
#         for i in range(50):
#             # print("-------------")
#             # print("第{}次测试".format(i))
#             predict = model(user_item_tensor2d, user_item_tensor2d.T, u_i_dict, l)
#             mse_loss = mse(predict, target)
#             rmse_loss = math.sqrt(mse_loss)
#             output = torch.zeros((predict.shape[0], len(u_i_dict[0])))
#             for j in u_i_dict.keys():
#                 for k, v in enumerate(u_i_dict[j]):
#                     output[j][k] = predict[j][v]
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
#     # print("bast mrr:{}".format(max_mrr))
#         print("aver recall:{}".format(aver_recall_list_))
#         print("aver precision:{}".format(aver_pre_list_))
#         print("aver f1-score:{}".format(aver_f1_list_))
#         print("mse:{}".format(mse_loss))
#         print("rmse:{}".format(rmse_loss))
