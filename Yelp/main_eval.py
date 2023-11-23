import torch
from torch import nn
import numpy as np
import math
import cv2
import open3d as o3d
from sklearn.metrics.pairwise import cosine_similarity
import config
from model_yelp_cpu_test2 import NN
from utils import *
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
# model.load_state_dict(torch.load("./model/2022-12-27_19_52_41_model.pth"))
# model.load_state_dict(torch.load("./model/2023-06-16_19_28_12_model_0.1_60.pth"))
# model.load_state_dict(torch.load("./model/2023-01-16_23_11_32_model_0.01.pth"))
# model.load_state_dict(torch.load("./model/2023-01-16_23_07_51_model_0.005.pth"))

model.load_state_dict(torch.load("./model/2023-11-14_15_18_14_model_0.1_60.pth"))

mse = torch.nn.MSELoss()
model.eval()
u_i_dict = test_deal(dataset, test_data)

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
#             output, x_u, x_v = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_i_dict, v_v_dict)
#             N = 1
#             hits_list = []
#             # ndcg_aver_list = []
#             # mrr_list = []
#             recall_list = []
#             precision_list = []
#             f1_list = []
#             for j in range(20):
#                 hits_, hits_dict = hits(u_i_dict, output, N)
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

# user_join_poi_dict = {}
# for i in range(target.shape[0]):
#     join_index_list = []
#     for j in range(target.shape[1]):
#         if target[i][j] != 0:
#             join_index_list.append(j)
#     if len(join_index_list) >= 5:
#         user_join_poi_dict[i] = join_index_list
# print(user_join_poi_dict)

# with torch.no_grad():
#     output, x_u_ts, x_v_ts = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_u_dict, v_v_dict)
#
#     test_topn_list2d = []
#     output_ = output.detach().numpy()
#     for k in u_i_dict.keys():
#         topn_list1d = []
#         tsne_list1d = []
#         for j in u_i_dict[k]:
#             topn_list1d.append(output_[k][j])
#             # tsne_list1d.append(x_v_ts[j])
#         test_topn_list2d.append(topn_list1d)
#         # tsne_list2d.append(tsne_list1d)
#     test_topn_arr2d = np.array(test_topn_list2d)
#     j = 5
#     hits_, hits_dict = hits(u_i_dict, test_topn_arr2d, j)
#     user_list = list(hits_dict.keys())
#     hits_user_list = []
#     for i in range(len(hits_)):
#         if hits_[i] == 1:
#             hits_user_list.append(user_list[i])
#     # 过滤到数组中为0的元素
#     user_join_item_list, user_join_item_index_list2d = [], []
#     for i in range(len(hits_user_list)):
#         user_join_item_list.append(list(filter(lambda x: x != 0, target[hits_user_list[i]].tolist())))
#         user_join_item_index_list = []
#         for j in range(len(output_[0])):
#             if target[hits_user_list[i]][j] != 0:
#                 user_join_item_index_list.append(j)
#         user_join_item_index_list2d.append(user_join_item_index_list)
#
#     # 预测
#     test_user_join_item_list2d = []
#     for i in range(len(user_join_item_index_list2d)):
#         test_user_join_item_list = []
#         for j in range(len(user_join_item_index_list2d[i])):
#             test_user_join_item_list.append(output_[hits_user_list[i]][j])
#         test_user_join_item_list2d.append(test_user_join_item_list)
#     # print(output)
#     # 过滤掉长度<=2
#     new_user_join_item_list2d, new_test_user_join_item_list2d = [], []
#     for i in range(len(user_join_item_list)):
#         if len(user_join_item_list[i]) >= 3:
#             new_user_join_item_list2d.append(user_join_item_list[i])
#             new_test_user_join_item_list2d.append(test_user_join_item_list2d[i])
#     user_join_item_list3d = []
#     user_join_item_list3d.append(new_user_join_item_list2d)
#     user_join_item_list3d.append(new_test_user_join_item_list2d)
#     # 绘制颜色矩阵
#     draw_color_matrix(user_join_item_list3d, x_title="POI", y_title="User", x_dis=len(new_user_join_item_list2d),
#                       y_dis=len(user_join_item_list3d),
#                       save_path="../images/plt/train_user_poi.png",
#                       display_ticks=True)
#     # draw_color_matrix(new_test_user_join_item_list2d, x_title="POI", y_title="User",
#     #                   x_dis=len(new_test_user_join_item_list2d),
#     #                   y_dis=1,
#     #                   save_path="../images/plt/test_user_poi.png",
#     #                   display_ticks=True)
#     equal = 0
#     # for i in range(len(test_user_join_item_list2d)):
#     #     if len(test_user_join_item_list2d[i]) == len(user_join_item_list[i]):
#     #         equal += 1
#     print(equal)

# with torch.no_grad():
#     # for q in range(100):
#     #     print("-------------")
#     #     aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
#     for i in range(10):
#         output, x_u_ts, x_v_ts = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_i_dict, v_v_dict)
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
#     # 获得高阶poi邻居
#     # user_item_jump_list = []
#     # user_item_jump_list.append(u_u_dict[user_item_keys_list[hits_user_index]])
#     # user_item_jump_list = get_jump_poi(user_item_keys_list[hits_user_index], u_u_dict, ug_v_v2[0], 2)
#     # user_item_jump_list = get_jump_poi2(user_item_keys_list[hits_user_index], u_u_dict, ug_v_v2[0], [], 3, 0)
#     # 获得命中用户的用户邻居
#     # for i in range(len(ug_u_u2[0])):
#     #     if ug_u_u2[0][i] == user_item_keys_list[hits_user_index]:
#     #         if (i+1)%2 == 0:
#     #             hits_user_user_neibor.append(ug_u_u2[0][i])
#     #         else:
#     #             hits_user_user_neibor.append(ug_u_u2[0][i+1])
#     #
#     # # 获得poi邻居的poi邻居
#     # hits_user_poi_poi_neibor, hits_user_poi_poi_neibor2d = [], []
#     # for i in range(len(hits_user_item_neibor)):
#     #     test = []
#     #     for j in range(len(ug_v_v2[0])):
#     #         if ug_v_v2[0][j] == hits_user_item_neibor[i]:
#     #             if (j+1) % 2 == 0:
#     #                 hits_user_poi_poi_neibor.append(ug_v_v2[0][j])
#     #                 test.append(ug_v_v2[0][j])
#     #             else:
#     #                 hits_user_poi_poi_neibor.append(ug_v_v2[0][j+1])
#     #                 test.append(ug_v_v2[0][j+1])
#     #     hits_user_poi_poi_neibor2d.append(test)
#     #
#     # # 获得poi邻居的poi邻居的poi邻居
#     # hits_user_poi_poi_poi_neibor = []
#     # for i in range(len(hits_user_poi_poi_neibor2d)):
#     #     test2d = []
#     #     for j in range(len(hits_user_poi_poi_neibor2d[i])):
#     #         test = []
#     #         for k in range(len(ug_v_v2[0])):
#     #             if ug_v_v2[0][k] == hits_user_poi_poi_neibor2d[i][j]:
#     #                 if (k + 1) % 2 == 0:
#     #                     # hits_user_poi_poi_neibor.append(ug_v_v2[0][k])
#     #                     test.append(ug_v_v2[0][k])
#     #                 else:
#     #                     # hits_user_poi_poi_neibor.append(ug_v_v2[0][k + 1])
#     #                     test.append(ug_v_v2[0][k + 1])
#     #         test2d.append(test)
#     #     hits_user_poi_poi_poi_neibor.append(test2d)
#     #
#     # # 获得poi邻居的poi邻居的用户邻居
#     # hits_user_poi_poi_user_neibor = []
#     # for i in range(len(hits_user_poi_poi_neibor2d)):
#     #     test2d = []
#     #     for j in range(len(hits_user_poi_poi_neibor2d[i])):
#     #         if len(v_v_dict[hits_user_poi_poi_neibor2d[i][j]]) != 0:
#     #             test = v_v_dict[hits_user_poi_poi_neibor2d[i][j]]
#     #         else:
#     #             test = []
#     #         test2d.append(test)
#     #     hits_user_poi_poi_user_neibor.append(test2d)
#     #
#     # # 获得poi邻居的user邻居
#     # hits_user_poi_user_neibor = []
#     # for i in range(len(hits_user_item_neibor)):
#     #     if len(v_v_dict[hits_user_item_neibor[i]]) != 0:
#     #         hits_user_poi_user_neibor.append(v_v_dict[hits_user_item_neibor[i]])
#     #     else:
#     #         hits_user_poi_user_neibor.append([])
#     #
#     # # 获得poi邻居的user邻居的poi邻居
#     # hits_user_poi_user_poi_neibor = []
#     # for i in range(len(hits_user_poi_user_neibor)):
#     #     test2d = []
#     #     for j in range(len(hits_user_poi_user_neibor[i])):
#     #         if len(u_u_dict[hits_user_poi_user_neibor[i][j]]) != 0:
#     #             test = u_u_dict[hits_user_poi_user_neibor[i][j]]
#     #         else:
#     #             test = []
#     #         test2d.append(test)
#     #     hits_user_poi_user_poi_neibor.append(test2d)
#     #
#     # # 获得poi邻居的user邻居的user邻居
#     # hits_user_poi_user_user_neibor = []
#     # for i in range(len(hits_user_poi_user_neibor)):
#     #     test2d = []
#     #     for j in range(len(hits_user_poi_user_neibor[i])):
#     #         test = []
#     #         for k in range(len(ug_u_u2[0])):
#     #             if ug_u_u2[0][k] == hits_user_poi_user_neibor[i][j]:
#     #                 if (k + 1) % 2 == 0:
#     #                     # hits_user_poi_poi_neibor.append(ug_u_u2[0][k])
#     #                     test.append(ug_u_u2[0][k])
#     #                 else:
#     #                     # hits_user_poi_poi_neibor.append(ug_u_u2[0][k + 1])
#     #                     test.append(ug_u_u2[0][k + 1])
#     #         test2d.append(test)
#     #     hits_user_poi_user_user_neibor.append(test2d)
#     #
#     # # 获得user邻居的user邻居
#     # hits_user_user_user_neibor, hits_user_user_user_neibor2d = [], []
#     # for i in range(len(hits_user_user_neibor)):
#     #     test = []
#     #     for j in range(len(ug_u_u2[0])):
#     #         if ug_u_u2[0][j] == hits_user_user_neibor[i]:
#     #             if (j+1) % 2 == 0:
#     #                 hits_user_user_user_neibor.append(ug_u_u2[0][j])
#     #                 test.append(ug_u_u2[0][j])
#     #             else:
#     #                 hits_user_user_user_neibor.append([ug_u_u2[0][j+1]])
#     #                 test.append(ug_u_u2[0][j+1])
#     #     hits_user_user_user_neibor2d.append(test)
#     #
#     # # 获得user邻居的user邻居的poi邻居
#     # hits_user_user_user_poi_neibor = []
#     # for i in range(len(hits_user_user_user_neibor2d)):
#     #     test2d = []
#     #     for j in range(len(hits_user_user_user_neibor2d[i])):
#     #         if len(u_u_dict[hits_user_user_user_neibor2d[i][j]]) != 0:
#     #             test = u_u_dict[hits_user_user_user_neibor2d[i][j]]
#     #         else:
#     #             test = []
#     #         test2d.append(test)
#     #     hits_user_user_user_poi_neibor.append(test2d)
#     #
#     # # 获得user邻居的user邻居的user邻居
#     # hits_user_user_user_user_neibor = []
#     # for i in range(len(hits_user_user_user_neibor2d)):
#     #     test2d = []
#     #     for j in range(len(hits_user_user_user_neibor2d[i])):
#     #         test = []
#     #         for k in range(len(ug_u_u2[0])):
#     #             if ug_u_u2[0][k] == hits_user_user_user_neibor2d[i][j]:
#     #                 if (k + 1) % 2 == 0:
#     #                     # hits_user_user_user_neibor.append(ug_u_u2[0][k])
#     #                     test.append(ug_u_u2[0][k])
#     #                 else:
#     #                     # hits_user_user_user_neibor.append([ug_u_u2[0][k + 1]])
#     #                     test.append(ug_u_u2[0][k + 1])
#     #         test2d.append(test)
#     #     hits_user_user_user_user_neibor.append(test2d)
#     #
#     # # 获得user邻居的poi邻居
#     # hits_user_user_poi_neibor = []
#     # for i in range(len(hits_user_user_neibor)):
#     #     if len(u_u_dict[hits_user_user_neibor[i]]) != 0:
#     #         hits_user_user_poi_neibor.append(u_u_dict[hits_user_user_neibor[i]])
#     #     else:
#     #         hits_user_user_poi_neibor.append([])
#     #
#     # # 获得user邻居的poi邻居的user邻居
#     # hits_user_user_poi_user_neibor = []
#     # for i in range(len(hits_user_user_poi_neibor)):
#     #     test2d = []
#     #     for j in range(len(hits_user_user_poi_neibor[i])):
#     #         if len(v_v_dict[hits_user_user_poi_neibor[i][j]]) != 0:
#     #             test = v_v_dict[hits_user_user_poi_neibor[i][j]]
#     #         else:
#     #             test = []
#     #         test2d.append(test)
#     #     hits_user_user_poi_user_neibor.append(test2d)
#     #
#     # # 获得user邻居的poi邻居的poi邻居
#     # hits_user_user_poi_poi_neibor = []
#     # for i in range(len(hits_user_user_poi_neibor)):
#     #     test2d = []
#     #     for j in range(len(hits_user_user_poi_neibor[i])):
#     #         test = []
#     #         for k in range(len(ug_v_v2[0])):
#     #             if ug_v_v2[0][k] == hits_user_user_poi_neibor[i][j]:
#     #                 if (k + 1) % 2 == 0:
#     #                     # hits_user_user_user_neibor.append(ug_u_u2[0][k])
#     #                     test.append(ug_v_v2[0][k])
#     #                 else:
#     #                     # hits_user_user_user_neibor.append([ug_u_u2[0][k + 1]])
#     #                     test.append(ug_v_v2[0][k + 1])
#     #         test2d.append(test)
#     #     hits_user_user_poi_poi_neibor.append(test2d)
#
#     # 获得top-k poi的得分
#         top_k_item_score_list = []
#         for i in range(len(top_k_item_list)):
#             top_k_item_score_list.append(round(output_[user_item_keys_list[hits_user_index]][top_k_item_list[i]], 4))
#         print("=================")
#     # print(user_item_keys_list[hits_user_index])
#     # print("hits_user_item_neibor", hits_user_item_neibor)
#     # print("hits_user_user_neibor", hits_user_user_neibor)
#     # print("hits_user_poi_poi_neibor", hits_user_poi_poi_neibor2d)
#     # print("hits_user_poi_user_neibor", hits_user_poi_user_neibor)
#     # print("hits_user_user_user_neibor", hits_user_user_user_neibor2d)
#     # print("hits_user_user_poi_neibor", hits_user_user_poi_neibor)
#     # print("hits_user_poi_poi_user_neibor", hits_user_poi_poi_user_neibor)
#     # print("hits_user_poi_poi_poi_neibor", hits_user_poi_poi_poi_neibor)
#     # print("hits_user_poi_user_user_neibor", hits_user_poi_user_user_neibor)
#     # print("hits_user_poi_user_poi_neibor", hits_user_poi_user_poi_neibor)
#     # print("hits_user_user_user_user_neibor", hits_user_user_user_user_neibor)
#     # print("hits_user_user_user_poi_neibor", hits_user_user_user_poi_neibor)
#     # print("hits_user_user_poi_user_neibor", hits_user_user_poi_user_neibor)
#     # print("hits_user_user_poi_poi_neibor", hits_user_user_poi_poi_neibor)
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
#
# # 余弦相似性
# for i in range(len(hits_user_item_list2d)):
#     if i == 0:
#         hits_user_item_cos = cosine_similarity(hits_user_embed.detach().numpy().reshape(1, -1),
#                                                np.array(hits_user_item_list2d[i]).reshape(1, -1))
#     else:
#         hits_user_item_cos = np.concatenate(
#             (hits_user_item_cos, cosine_similarity(hits_user_embed.detach().numpy().reshape(1, -1),
#                                                    np.array(hits_user_item_list2d[i]).reshape(1, -1))), axis=0)
# cosine_sort_index = np.argsort(np.squeeze(hits_user_item_cos))
# 实现余弦相似性散点图
# drawer("cosine", "point", "cosine", [1 for i in range(len(cosine_sort_index))],
#        np.squeeze(hits_user_item_cos).tolist(),
#        "POI", None, hits_user_item_list)
# 得分散点图
# 命中用户的POI得分
# hits_user_item_score = output_[user_item_keys_list[hits_user_index]].tolist()
# hits_user_item_score_list = []
# for i in range(len(hits_user_item_list)):
#     hits_user_item_score_list.append(hits_user_item_score[hits_user_item_list[i]])
# drawer("score", "point", "score", [1 for i in range(len(cosine_sort_index))], hits_user_item_score_list, "POI",
#        None, hits_user_item_list)
# print(hits_user_item_score_list)
# print(cosine_sort_index)
# hits_user_item_cos = hits_user_item_cos - hits_user_item_cos.min()
# hits_user_item_cos = hits_user_item_cos / (hits_user_item_cos.max() + 1e-8)
# heatmap = cv2.applyColorMap(np.uint8(255 * hits_user_item_cos), cv2.COLORMAP_JET)
# heatmap = cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB)
# heatmap = heatmap.squeeze() / 255.0
# pointcloud = o3d.geometry.PointCloud()
# pointcloud.points = o3d.utility.Vector3dVector(heatmap)
# o3d.io.write_point_cloud('./temp/sim_{}_pcd.ply'.format("cos"), pointcloud)
# o3d.visualization.draw_geometries([pointcloud])

# cosine_similarity(hits_user_embed.detach().numpy())
# ndcg_ = get_ndcg(hits_dict, u_i_dict)
# ndcg_aver_list.append(ndcg_)
# mrr_ = mrr(hits_dict, u_i_dict)
# mrr_list.append(mrr_)
#         recall_list.append(recall_topn(hits_))
#         precision_list.append(precision_topn(hits_, N))
#         f1_list.append(f1_score_topn(recall_topn(hits_), precision_topn(hits_, N)))
#
#         N += 1
#
#     aver_recall_list.append(recall_list)
#     aver_pre_list.append(precision_list)
#     aver_f1_list.append(f1_list)
#
# aver_recall_list_, aver_pre_list_, aver_f1_list_ = [], [], []
# for i in range(len(aver_recall_list[0])):
#     recall_sum, pre_sum, f1_sum = 0, 0, 0
#     for j in range(len(aver_recall_list)):
#         recall_sum += aver_recall_list[j][i]
#         pre_sum += aver_pre_list[j][i]
#         f1_sum += aver_f1_list[j][i]
#     aver_recall_list_.append(round(recall_sum / len(aver_recall_list), 5))
#     aver_pre_list_.append(round(pre_sum / len(aver_pre_list), 5))
#     aver_f1_list_.append(round(f1_sum / len(aver_f1_list), 5))
#
# print("aver recall:{}".format(aver_recall_list_))
# print("aver precision:{}".format(aver_pre_list_))
# print("aver f1-score:{}".format(aver_f1_list_))
#
# print("mse:{}".format(mse_loss))
# print("rmse:{}".format(rmse_loss))
