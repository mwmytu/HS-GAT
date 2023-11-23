import time
import torch
from torch import nn
import numpy as np

import config
from model_bpr import NN
from utils import delete_stopwords, \
    cut_words_key, \
    get_glove, charToEmbed
from dataloader import getData, train_deal
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

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(100):
    model.train()
    start_time = time.time()
    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()

    train_loss_sum = []
    train_loss_sum2 = []
    for user, item_i, item_j in train_dataset.ng_sample():
        model.zero_grad()
        prediction_i, prediction_j, loss, loss2 = model(word_embed, u_v_matrix, v_u_matrix, ug_v_v2, ug_u_u2, v_v_dict,
                                                        user, item_i, item_j)
        loss.backward()
        optimizer.step()
        count += 1
        train_loss_sum.append(loss.item())
        train_loss_sum2.append(loss2.item())
        # print(count)

    elapsed_time = time.time() - start_time
    train_loss = round(np.mean(train_loss_sum[:-1]), 4)  # 最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)  # 最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
        train_loss) + "=" + str(train_loss2) + "+"
    print('--train--', elapsed_time)

nowDate = time.strftime("%Y-%m-%d_%H_%M_%S")
torch.save(model.state_dict(), "./model/LR-GCCF-" + nowDate + ".pth")
