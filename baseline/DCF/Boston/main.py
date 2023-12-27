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

dataset, train_data, test_data = getData()
target, u_u_dict, v_v_dict = train_deal(dataset, train_data)
user_item_matrix = get_matrix_factor(dataset, train_data)
user_item_tensor2d = torch.tensor(user_item_matrix, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)
# 获得模型
model = DCF(user_item_tensor2d, user_item_tensor2d.T)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
l = 0
for k in u_u_dict.keys():
    for j in u_u_dict[k]:
        l += 1
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
    predict = model(user_item_tensor2d, user_item_tensor2d.T, u_u_dict, l)
    # predict = predict.float()
    # label, predict = model()
    regularization_loss = 0
    # for name, parameters in model.state_dict().items():
    #     if "weight" in name:
    #         print(name, ':', parameters.detach().numpy())
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        #     regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)

    # loss = mse(target, torch.argmax(predict, 1).float())
    loss = mse(predict, target) + 0.000001 * regularization_loss
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
torch.save(model.state_dict(), "./model/DCF-" + nowDate + ".pth")
