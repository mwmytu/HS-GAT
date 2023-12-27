import datetime
from torch import nn
import math

import config
from utils import *
from dataloader2 import getData, train_deal, test_deal
from model import ATST_LSTM

# 指定gpu设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ratings, train_user_matrix, train_event_matrix = data_prepare.data_partition(fname='./data')
dataset, train_data, test_data, ug_u_u2, ug_v_v2 = getData()
u_v_matrix, v_u_matrix, target, words_list, u_u_dict, v_v_dict, u_u_dict_all = train_deal(dataset, train_data)

model = ATST_LSTM(
    u_v_matrix.shape[1],
    v_u_matrix.shape[1],
    config.n_emb,
    config.n_emb,
    config.dense_hidden1,
    config.dense_hidden2,
    config.dense_out,
    u_v_matrix,
    config.at_header_num
)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
mse = nn.MSELoss(reduction='mean')
target = torch.tensor(target, dtype=torch.float32)
for i in range(config.epochs):
    optimizer.zero_grad()
    predict = model(u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2)
    loss = mse(predict, target)
    print("loss:{}".format(loss.item()))
    loss.backward()
    optimizer.step()

time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

torch.save(model.state_dict(), "./models/" + str(time) + ".pth")
