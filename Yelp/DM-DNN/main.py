import datetime
from torch import nn
import math

import config
from model_yelp_cpu_test2 import NN
from utils import *
from dataloader import getData, train_deal

# 指定gpu设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ratings, train_user_matrix, train_event_matrix = data_prepare.data_partition(fname='./data')
dataset, train_data, test_data, ug_u_u2, ug_v_v2 = getData()
u_v_matrix, v_u_matrix, target, words_list, u_u_dict, v_v_dict, u_u_dict_all = train_deal(dataset, train_data)
# 对target使用计算权重进行处理
# class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(target), y=target)
# class_weights = torch.tensor(class_weights, dtype=torch.float)
target = torch.tensor(target, dtype=torch.float32)
# target = torch.squeeze(target.reshape((-1, 1)))
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    predict = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_u_dict, v_v_dict)
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

    # if max_acc < acc:
    #     max_acc = acc
print(acc_list)
x = [i for i in range(epochs)]
time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
# 绘制损失曲线
# drawer(x, loss_list, "epochs", "loss", "损失值曲线", time + "_loss.jpg")
# # 绘制准确率曲线
# drawer(x, acc_list, "epochs", "acc", "准确率曲线", time + "_acc.jpg")

torch.save(model.state_dict(), "./model/" + str(time) + "_model.pth")

# 转为测试
# model.eval()
# u_i_dict = test_deal(dataset, test_data)
#
# with torch.no_grad():
#     output = model(word_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_i_dict, v_v_dict)
#
# hits = []
# predicted_labels = output.detach().numpy()
# predicted_labels = predicted_labels.reshape((len(u_i_dict), 100, 2))
# for k, v in enumerate(u_i_dict):
#     top20_items = [u_i_dict[v][i] for i in np.argsort(predicted_labels[v])[::-1][0:20]]
#     if v in top20_items:
#         hits.append(1)
#     else:
#         hits.append(0)
#
# # precN = precision_topn(hits)
# # recallN = recall_topn(hits, 10)
#
# print("The Hit Ratio @ 20 is {:.2f}".format(np.average(hits)))
# print("The precision @ 10 is {:.2f}".format(precN))
# print("The recall @ 10 is {:.2f}".format(recallN))
