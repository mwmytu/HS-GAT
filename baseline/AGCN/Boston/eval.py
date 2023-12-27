import time
import torch
import math
import datetime
from torch import nn
import numpy as np
import pandas as pd
import math
from sklearn.utils.class_weight import compute_class_weight

from dataloader import get_data, train_deal, test_deal, get_matrix_factor
from model import AGCN
from config import epochs
import random
from utils import hits, recall_topn, precision_topn, f1_score_topn

# 获得数据
dataset, train_data, test_data, u_v_arr2d_, ug_u_u2, ug_v_v2 = get_data()
user_item_matrix = get_matrix_factor(dataset, train_data)
uv_tag, u_u_dict, train_data, u_u_dict_all = train_deal(dataset, train_data)
# target
target = torch.tensor(user_item_matrix, dtype=torch.float32)
# 目标
user_item_tensor2d = torch.tensor(user_item_matrix, dtype=torch.int32)

# 获得模型
mse = torch.nn.MSELoss()
model = AGCN(user_item_tensor2d, user_item_tensor2d.T)
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=["acc"])
model.load_state_dict(torch.load("./model/AGCN-2022-12-17_13_58_20.pth"))

u_i_dict = test_deal(dataset, test_data)

with torch.no_grad():
    for q in range(50):
        print("-------------")
        max_hits, max_recall, max_precision, max_f1 = 0, 0, 0, 0
        # max_mrr = 0
        max_hits_sample = 0
        # min_mrr_sample = 0
        max_hits_list = [0 for i in range(20)]
        max_recall_list = [0 for i in range(20)]
        max_pre_list = [0 for i in range(20)]
        max_f1_list = [0 for i in range(20)]
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):

            # print("第{}次测试".format(i))
            predict = model(user_item_tensor2d, user_item_tensor2d.T, ug_u_u2, ug_v_v2)
            output = torch.zeros((predict.shape[0], len(u_i_dict[0])))
            for j in u_i_dict.keys():
                for k, v in enumerate(u_i_dict[j]):
                    output[j][k] = predict[j][v]
            mse_loss = mse(predict, target)
            rmse_loss = math.sqrt(mse_loss)
            N = 1
            hits_list = []
            # ndcg_aver_list = []
            # mrr_list = []
            recall_list = []
            precision_list = []
            f1_list = []
            for j in range(20):
                hits_ = hits(u_i_dict, output, N)
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

            for j in range(20):
                if max_recall_list[j] < recall_list[j]:
                    max_recall_list[j] = recall_list[j]
                if max_pre_list[j] < precision_list[j]:
                    max_pre_list[j] = precision_list[j]
                if max_f1_list[j] < f1_list[j]:
                    max_f1_list[j] = f1_list[j]

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

    # print("bast mrr:{}".format(max_mrr))
    #     print("bast recall:{}".format(max_recall_list))
    #     print("bast precision:{}".format(max_pre_list))
    #     print("bast f1-score:{}".format(max_f1_list))
        print("aver recall:{}".format(aver_recall_list_))
        print("aver precision:{}".format(aver_pre_list_))
        print("aver f1-score:{}".format(aver_f1_list_))

        print("mse:{}".format(mse_loss))
        print("rmse:{}".format(rmse_loss))
