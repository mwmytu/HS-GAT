import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import math
import datetime
import config
from sklearn.preprocessing import normalize, MinMaxScaler
import torch.nn.functional as F

from dataloader import getData, train_deal, test_deal
from utils import hits, recall_topn, precision_topn, f1_score_topn
from model import MLP
from config import n_emb, epochs


# 获得数据集
dataset, train_data, test_data = getData()
user_id, item_id, target, u_u_dict = train_deal(dataset, train_data)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
user_id = torch.tensor(user_id)
item_id = torch.tensor(item_id)
num_users = len(user_id)
num_items = len(item_id)

model = MLP(user_id, item_id, n_emb)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss(reduction="mean")

for i in range(epochs):
    optimizer.zero_grad()
    output = model(user_id, item_id, u_u_dict)
    loss = mse(output, target)

    print("=====第{}次迭代=====".format(i + 1))
    print("MSE Loss:{}".format(loss.item()))
    print("RMSE Loss:{}".format(math.sqrt(loss.item())))
    print("=======******=======")

    loss.backward()
    optimizer.step()

x = [i for i in range(epochs)]
time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

torch.save(model.state_dict(), "./model/" + str(time) + "_model.pth")
