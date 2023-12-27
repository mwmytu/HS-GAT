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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dense
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
# for e in range(10):
for i in range(epochs):
    # optimizer.zero_grad()
    # loss = model()
    # model.train()
    optimizer.zero_grad()
    # predict = model()
    predict = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2, u_u_dict)
    # predict = predict.float()
    # label, predict = model()

    # loss = mse(target, torch.argmax(predict, 1).float())
    loss = mse(predict, target)
    mae_loss = mae(predict, target)

    # acc = accuracy(target, predict)

    # loss_ = loss.item()
    # loss_list.append(loss.item())
    # acc_list.append(acc)
    print("=====第{}次迭代=====".format(i + 1))
    print("MSE Loss:{}".format(loss.item()))
    print("RMSE Loss:{}".format(math.sqrt(loss.item())))
    print("MAE Loss:{}".format(math.sqrt(mae_loss.item())))
    # print("acc:{}".format(acc))
    print("=======******=======")
    # mse.requires_grad_(True)
    # criteria.requires_grad_(True)
    loss.backward()
    optimizer.step()

# BPR
# print('--------training processing-------')
# count, best_hr = 0, 0
# for epoch in range(5):
#     model.train()
#     start_time = time.time()
#     # pdb.set_trace()
#     print('train data of ng_sample is  end')
#     # elapsed_time = time.time() - start_time
#     # print(' time:'+str(round(elapsed_time,1)))
#     # start_time = time.time()
#
#     train_loss_sum = []
#     train_loss_sum2 = []
#     for user, item_i, item_j in train_dataset.ng_sample():
#         model.zero_grad()
#         prediction_i, prediction_j, loss, loss2 = model(u_v_tensor1d, user_tensor1d, item_tensor1d, ug_u_u2, ug_v_v2,
#                                                         user, item_i, item_j)
#         loss.backward()
#         optimizer.step()
#         count += 1
#         train_loss_sum.append(loss.item())
#         train_loss_sum2.append(loss2.item())
#         # print(count)
#
#     elapsed_time = time.time() - start_time
#     train_loss = round(np.mean(train_loss_sum[:-1]), 4)  # 最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
#     train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)  # 最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
#     str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
#         train_loss) + "=" + str(train_loss2) + "+"
#     print('--train--', elapsed_time)
#
nowDate = time.strftime("%Y-%m-%d_%H_%M_%S")
torch.save(model.state_dict(), "./model/FG-CF-" + nowDate + ".pth")
