import time
import torch
import datetime
from torch import nn
import numpy as np
import pandas as pd
import math
from sklearn.utils.class_weight import compute_class_weight

from dataloader import get_data, train_deal, test_deal
from model import SEMA
from config import epochs

dataset, train_data, test_data = get_data()
user_item_word_list, item_user_word_list, target, u_u_dict, train_data, user_latent_factors, item_latent_factors = train_deal(
    dataset, train_data)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
user_item_word_list_, item_user_word_list_ = [], []
for i in range(len(user_item_word_list)):
    for j in [int(j) for j in sum(user_item_word_list[i], [])]:
        user_item_word_list_.append(j)
for i in range(len(item_user_word_list)):
    for j in [int(j) for j in sum(item_user_word_list[i], [])]:
        item_user_word_list_.append(j)

user_item_words_df = pd.DataFrame(user_item_word_list_, columns=["words"])
item_user_words_df = pd.DataFrame(item_user_word_list_, columns=["words"])

user_item_words_df.to_csv("./word_data/user_item_words.csv", encoding='utf8')
item_user_words_df.to_csv("./word_data/item_user_words.csv", encoding="utf8")

user_dim = len(user_item_word_list_)
item_dim = len(item_user_word_list_)
user_num = len(user_item_word_list)
item_num = len(item_user_word_list)

user_item_word_tensor = torch.tensor(user_item_word_list_)
item_user_word_tensor = torch.tensor(item_user_word_list_)
user_latent_factors_tensor = torch.tensor(user_latent_factors)
item_latent_factors_tensor = torch.tensor(item_latent_factors)

model = SEMA(user_item_word_tensor, item_user_word_tensor, user_dim, item_dim, user_num, item_num,
             user_latent_factors_tensor, item_latent_factors_tensor)
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=["acc"])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss(reduction='mean')
mae = nn.L1Loss(reduction='mean')
for i in range(epochs):
    optimizer.zero_grad()
    score = model(user_item_word_tensor, item_user_word_tensor, user_latent_factors_tensor, item_latent_factors_tensor,
                  u_u_dict)
    loss = mse(target, score)
    rmse_loss = math.sqrt(loss)
    print("======第{}次迭代======".format(i + 1))
    print("MSE Loss:{0:5}".format(loss))
    print("RMSE Loss:{0:5}".format(rmse_loss))
    print("======迭代结束======")
    # loss.requires_grad_(True)
    loss.backward()
    optimizer.step()

nowDate = time.strftime("%Y-%m-%d_%H_%M_%S")
torch.save(model.state_dict(), "./model/SEMA-" + nowDate + ".pth")
# u_i_dict = test_deal(dataset, test_data)
# output = model.predict([user_item_word_tensor, item_user_word_tensor, u_i_dict])
