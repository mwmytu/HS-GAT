import matplotlib.pyplot as plt
import datetime
from math import pow
import numpy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_normal, normal_, zeros_, ones_


from dataloader import getData, train_deal, test_deal, get_matrix_factor
from utils import save_to_path
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

def matrix_factorization(R, P, Q, K, steps=7000, alpha=0.05, beta=0.02):
    Q = Q.T  # .T操作表示矩阵的转置
    result = []
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])  # .dot(P,Q) 表示矩阵内积
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        result.append(e)
        if e < 0.001:
            break
        print("第{}次迭代，loss：{}".format(step, e))
    return P, Q.T, result


if __name__ == '__main__':

    dataset, train_data, test_data = getData()
    user_item_matrix = get_matrix_factor(dataset, train_data)

    # 初始的mf
    N = len(user_item_matrix)
    M = len(user_item_matrix[0])
    # 用户和项目embed的维度
    K = 10

    P = numpy.random.rand(N, K)  # 随机生成一个 N行 K列的矩阵
    Q = numpy.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵

    nP, nQ, result = matrix_factorization(user_item_matrix, P, Q, K)
    # print("原始的评分矩阵R为：\n", user_item_matrix)
    # R_MF = numpy.dot(nP, nQ.T)
    # print("经过MF算法填充0处评分值后的评分矩阵R_MF为：\n", R_MF)
    #
    nP_list2d, nQ_list2d = [], []
    for i in range(nP.shape[0]):
        nP_list2d.append(",".join([str(j) for j in nP[i]]))
    for i in range(nQ.shape[0]):
        nQ_list2d.append(",".join([str(j) for j in nQ[i]]))
    nP_df = pd.DataFrame(nP_list2d, columns=["user_embed"])
    nQ_df = pd.DataFrame(nQ_list2d, columns=["item_embed"])
    save_to_path(nP_df, "./embed/user_embed_10_35000.csv", "csv")
    save_to_path(nQ_df, "./embed/item_embed_10_35000.csv", "csv")
    #
    # # -------------损失函数的收敛曲线图---------------
    #
    # n = len(result)
    # x = range(n)
    # plt.plot(x, result, color='r', linewidth=3)
    # plt.title("Convergence curve")
    # plt.xlabel("generation")
    # plt.ylabel("loss")
    # plt.show()

    # 使用矩阵分解进行测试
    # u_i_dict = test_deal(dataset, test_data)
    # test_sample = np.zeros((len(u_i_dict), len(u_i_dict[0])), dtype=np.float32)
    # for k, v in enumerate(u_i_dict):
    #     for i in range(len(u_i_dict[v])):
    #         test_sample[v][i] = R_MF[v][u_i_dict[v][i]]

    # 神经网络的mf
    # target = torch.tensor(user_item_matrix, dtype=torch.float32)
    # user_num = len(dataset["user_id"].unique())
    # item_num = len(dataset["business_id"].unique())
    # mse = nn.MSELoss(reduction="mean")
    # model = MF(user_num, item_num)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # users_tensor = torch.tensor(dataset["user_id"].unique().tolist())
    # items_tensor = torch.tensor(dataset["business_id"].unique().tolist())
    #
    # for i in range(epochs):
    #     optimizer.zero_grad()
    #     u_embed, v_embed, output = model(users_tensor, items_tensor)
    #     regularization_loss = 0
    #     for param in model.parameters():
    #         # TODO: you may implement L1/L2 regularization here
    #         # 使用L2正则项
    #         #     regularization_loss += torch.sum(abs(param))
    #         regularization_loss += torch.sum(param ** 2)
    #
    #     # loss = mse(target, torch.argmax(predict, 1).float())
    #     loss = mse(output, target) + 0.05 * regularization_loss
    #     # loss = mse(output, target)
    #     print("=====第{}次迭代=====".format(i + 1))
    #     print("MSE Loss:{}".format(loss.item()))
    #     print("=======******=======")
    #
    #     loss.backward()
    #     optimizer.step()
    #
    # x = [i for i in range(epochs)]
    # time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    #
    # torch.save(model.state_dict(), "./model/" + str(time) + "_model.pth")



