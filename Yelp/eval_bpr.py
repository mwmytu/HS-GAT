import torch
from torch import nn
import numpy as np
import random

import config
from model_bpr import NN
from utils import delete_stopwords, \
    cut_words_key, \
    get_glove, charToEmbed, precision_topn, recall_topn, hits, f1_score_topn
from dataloader import getData, train_deal, test_deal
import data_utils

# 指定gpu设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ratings, train_user_matrix, train_event_matrix = data_prepare.data_partition(fname='./data')
dataset, train_data, test_data, ug_u_u2, ug_v_v2 = getData()
u_v_matrix, v_u_matrix, target, words_list, u_u_dict, v_v_dict, u_u_dict_all = train_deal(dataset, train_data)
user_num = len(dataset["user_id"].unique())
item_num = len(dataset["business_id"].unique())
u_i_counters = 0
for i in u_u_dict.keys():
    for j in u_u_dict[i]:
        u_i_counters += u_i_counters

# 对target使用计算权重进行处理
# class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(target), y=target)
# class_weights = torch.tensor(class_weights, dtype=torch.float)
target = torch.tensor(target, dtype=torch.float32)
target = torch.squeeze(target.reshape((-1, 1)))
# target = target.to(device)

# 获得word的embed
# words = v_u_fea[['user_id', 'business_id', 'text']]
# 去除重复值
# words.drop_duplicates(subset=['event_id'], keep='first', inplace=True)
# 删除停用词
# content = words["text"].to_list()
# content = replace_char(content)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
train_dataset = data_utils.BPRData(
    train_dict=u_u_dict, num_item=item_num, num_ng=5, is_training=True, \
    data_set_count=u_i_counters, all_rating=u_u_dict_all)

po_npo_dataset = train_dataset.ng_sample()

testing_dataset_loss = data_utils.BPRData(
    train_dict=u_u_dict, num_item=item_num, num_ng=5, is_training=True, \
    data_set_count=u_i_counters, all_rating=u_u_dict_all)
mse = nn.MSELoss(reduction='mean')
mae = nn.L1Loss(reduction='mean')
# mse.requires_grad_(True)
# criteria = nn.CrossEntropyLoss()
# criteria2 = nn.NLLLoss()
loss_list = []
acc_list = []
roc_list = []
max_acc = 0
acc = 0

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

print('--------training processing-------')
with torch.no_grad():
    max_hits, max_recall, max_precision, max_f1 = 0, 0, 0, 0
    aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
    max_recall_list = [0 for i in range(20)]
    max_pre_list = [0 for i in range(20)]
    max_f1_list = [0 for i in range(20)]
    for i in range(5):
        print("-------------")
        print("第{}次测试".format(i))
        test_score_list = []
        k = 0
        u_i_list, u_i_score_list, u_i_list2d, u_i_score_list2d = [], [], [], []
        i_score_num = 0
        for user, item_i, item_j in user_brp_data2d:
            k += 1
            prediction_i, prediction_j, loss, loss2 = model(word_embed, u_v_matrix, v_u_matrix, ug_v_v2, ug_u_u2, v_v_dict,
                                                        user, item_i, item_j)
            # i_score_num += prediction_i
            # u_i_list.append(item_j)
            u_i_score_list.append(prediction_j)
            if k % 99 == 0:
                # u_i_list2d.append(u_i_list + [item_i])
                u_i_score_list2d.append(u_i_score_list + [prediction_i])
                u_i_list, u_i_score_list = [], []
                # i_score_num = 0
        output = torch.tensor(u_i_score_list2d)
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

        for j in range(20):
            if max_recall_list[j] < recall_list[j]:
                max_recall_list[j] = recall_list[j]
            if max_pre_list[j] < precision_list[j]:
                max_pre_list[j] = precision_list[j]
            if max_f1_list[j] < f1_list[j]:
                max_f1_list[j] = f1_list[j]

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

        # 随机采样baseline
        # hits_sample_list = []
        # mrr_sample_list = []
        # hits_sample_list_ = []
        # mrr_sample_list_ = []
        #
        # u_i_sample_dict = {}
        # for k, v in enumerate(u_i_dict):
        #     u_i_sample = random.sample(u_i_dict[v], 40)
        #     # u_i_sample_ = random.sample(u_i_dict[v], 20)
        #     u_i_sample_dict[v] = u_i_sample
        #
        #     if u_i_dict[v][-1] in u_i_sample:
        #         hits_sample_list.append(1)
        #     else:
        #         hits_sample_list.append(0)
        #
        # mrr_sample = mrr(u_i_sample_dict, u_i_dict)

        # hits_aver = []
        # recall_list = []
        # precision_list = []
        # for i in range(len(hits_list)):
        #     hits_aver.append(np.average(hits_list[i]))
        #     recall_list.append(recall_topn(hits_list[i]))

        # print("mrr:{}".format(mrr_sample))
        # print("hits:{}".format(hits_aver))
        # print("random_hits:{}".format(np.average(hits_sample_list)))
        # print("recall@N:{}".format(recall_list))
        # print("NDCG@N:{}".format(ndcg_aver_list))
        # print("MRR@N:{}".format(mrr_list))

        # if max_hits < hits_aver[3]:
        #     max_hits = hits_aver[3]
        # if max_hits_sample < np.average(hits_sample_list):
        #     max_hits_sample = np.average(hits_sample_list)

        # if max_mrr > mrr_sample:
        #     max_mrr = mrr_sample
        # 获得最好的recall
        # if max_recall < recall_list[1]:
        #     max_recall = recall_list[1]
        # # 获得最好的precision
        # if max_precision < precision_list[1]:
        #     max_precision = precision_list[1]
        # # 获得最好的f1
        # if max_f1 < f1_list[1]:
        #     max_f1 = f1_list[1]
        # if min_mrr_sample < np.average(hits_sample_list):
        #     min_mrr_sample = np.average(hits_sample_list)

print("bast recall:{}".format(max_recall_list))
print("bast precision:{}".format(max_pre_list))
print("bast f1-score:{}".format(max_f1_list))
print("aver recall:{}".format(aver_recall_list_))
print("aver precision:{}".format(aver_pre_list_))
print("aver f1-score:{}".format(aver_f1_list_))