import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import datetime
import time

from dataloader import get_data, train_deal, test_deal, get_matrix_factor
from utils import hits, recall_topn, precision_topn, f1_score_topn
from config import epochs
from model import AGCN


# 获得数据
dataset, train_data, test_data, u_v_arr2d_, ug_u_u2, ug_v_v2 = get_data()
user_item_matrix = get_matrix_factor(dataset, train_data)
uv_tag, u_u_dict, train_data, u_u_dict_all = train_deal(dataset, train_data)
# target
target = torch.tensor(user_item_matrix, dtype=torch.float32)
# 目标
user_item_tensor2d = torch.tensor(user_item_matrix, dtype=torch.int32)

# 获得模型
model = AGCN(user_item_tensor2d, user_item_tensor2d.T)
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
    predict = model(user_item_tensor2d, user_item_tensor2d.T, ug_u_u2, ug_v_v2)
    # predict = predict.float()
    # label, predict = model()
    # regularization_loss = 0
    # # for name, parameters in model.state_dict().items():
    # #     if "weight" in name:
    # #         print(name, ':', parameters.detach().numpy())
    # for param in model.parameters():
    #     # TODO: you may implement L1/L2 regularization here
    #     # 使用L2正则项
    #     #     regularization_loss += torch.sum(abs(param))
    #     regularization_loss += torch.sum(param ** 2)
    #
    # # loss = mse(target, torch.argmax(predict, 1).float())
    # loss = mse(predict, target) + 0.000000000001 * regularization_loss
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

nowDate = time.strftime("%Y-%m-%d_%H_%M_%S")
torch.save(model.state_dict(), "./model/AGCN-" + nowDate + ".pth")
