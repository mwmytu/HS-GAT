import matplotlib.pyplot as plt
import datetime
from math import pow
import math
import numpy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_normal, normal_, zeros_, ones_


from dataloader import getData, train_deal, test_deal, get_matrix_factor
from utils import save_to_path, hits, recall_topn, precision_topn, f1_score_topn
from config import n_out, epochs


class MF(nn.Module):
    def __init__(self, user_num, item_num):
        super(MF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.user_embedding = nn.Embedding(self.user_num, n_out)
        self.item_embedding = nn.Embedding(self.item_num, n_out)
        normal_(self.user_embedding.weight, std=0.1)
        normal_(self.item_embedding.weight, std=0.1)

    def forward(self, users, items):
        users_e = self.user_embedding(users)
        items_e = self.item_embedding(items)
        score = torch.matmul(users_e, items_e.T)
        return users_e, items_e, score


dataset, train_data, test_data = getData()
user_item_matrix = get_matrix_factor(dataset, train_data)
# 神经网络的mf
target = torch.tensor(user_item_matrix, dtype=torch.float32)
user_num = len(dataset["user_id"].unique())
item_num = len(dataset["business_id"].unique())
mse = nn.MSELoss(reduction="mean")
model = MF(user_num, item_num)
model.load_state_dict(torch.load("./model/2022-12-17_13_30_34_model.pth"))
users_tensor = torch.tensor(dataset["user_id"].unique().tolist())
items_tensor = torch.tensor(dataset["business_id"].unique().tolist())

model.eval()
u_i_dict = test_deal(dataset, test_data)

# 获得embed
# with torch.no_grad():
#     u_embed, v_embed, output = model(users_tensor, items_tensor)
#     u_embed = u_embed.detach().numpy()
#     v_embed = v_embed.detach().numpy()
#     nP_list2d, nQ_list2d = [], []
#     for i in range(u_embed.shape[0]):
#         nP_list2d.append(",".join([str(j) for j in u_embed[i]]))
#     for i in range(v_embed.shape[0]):
#         nQ_list2d.append(",".join([str(j) for j in v_embed[i]]))
#     nP_df = pd.DataFrame(nP_list2d, columns=["user_embed"])
#     nQ_df = pd.DataFrame(nQ_list2d, columns=["item_embed"])
#     save_to_path(nP_df, "./embed/user_embed_30_40000.csv", "csv")
#     save_to_path(nQ_df, "./embed/item_embed_30_40000.csv", "csv")

with torch.no_grad():
    for q in range(50):
        print("-------------")
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):
            user_embed, item_embed, output = model(users_tensor, items_tensor)
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

