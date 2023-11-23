import matplotlib.pyplot as plt
from math import pow
import numpy
import numpy as np

from dataloader import getData, train_deal, test_deal, get_matrix_factor


def matrix_factorization(R, P, Q, K, steps=600, alpha=0.05, beta=0.02):
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

    dataset, user_item_matrix, test_data = get_matrix_factor()

    N = len(user_item_matrix)
    M = len(user_item_matrix[0])
    # 用户和项目embed的维度
    K = 20

    P = numpy.random.rand(N, K)  # 随机生成一个 N行 K列的矩阵
    Q = numpy.random.rand(M, K)  # 随机生成一个 M行 K列的矩阵

    nP, nQ, result = matrix_factorization(user_item_matrix, P, Q, K)
    print("原始的评分矩阵R为：\n", user_item_matrix)
    R_MF = numpy.dot(nP, nQ.T)
    print("经过MF算法填充0处评分值后的评分矩阵R_MF为：\n", R_MF)

    # -------------损失函数的收敛曲线图---------------

    n = len(result)
    x = range(n)
    plt.plot(x, result, color='r', linewidth=3)
    plt.title("Convergence curve")
    plt.xlabel("generation")
    plt.ylabel("loss")
    plt.show()

    # 使用矩阵分解进行测试
    u_i_dict = test_deal(dataset, test_data)
    test_sample = np.zeros((len(u_i_dict), len(u_i_dict[0])), dtype=np.float32)
    for k, v in enumerate(u_i_dict):
        for i in range(len(u_i_dict[v])):
            test_sample[v][i] = R_MF[v][u_i_dict[v][i]]

    hits_10, hits_20, hits_30, hits_40 = [], [], [], []
    for k, v in enumerate(u_i_dict):
        topn_items_10, topn_items_20, topn_items_30, topn_items_40 = [], [], [], []
        items_index_10 = np.argsort(np.array(test_sample[v]))[::-1][0:10]
        items_index_20 = np.argsort(np.array(test_sample[v]))[::-1][0:20]
        items_index_30 = np.argsort(np.array(test_sample[v]))[::-1][0:30]
        items_index_40 = np.argsort(np.array(test_sample[v]))[::-1][0:40]
        for i in range(len(items_index_10)):
            topn_items_10.append(u_i_dict[v][items_index_10[i]])
        for i in range(len(items_index_20)):
            topn_items_20.append(u_i_dict[v][items_index_20[i]])
        for i in range(len(items_index_30)):
            topn_items_30.append(u_i_dict[v][items_index_30[i]])
        for i in range(len(items_index_40)):
            topn_items_40.append(u_i_dict[v][items_index_40[i]])
        if u_i_dict[v][-1] in topn_items_10:
            hits_10.append(1)
        else:
            hits_10.append(0)
        if u_i_dict[v][-1] in topn_items_20:
            hits_20.append(1)
        else:
            hits_20.append(0)
        if u_i_dict[v][-1] in topn_items_30:
            hits_30.append(1)
        else:
            hits_30.append(0)
        if u_i_dict[v][-1] in topn_items_40:
            hits_40.append(1)
        else:
            hits_40.append(0)

    print("matrix_factorization HR@10:{}".format(np.average(hits_10)))
    print("matrix_factorization HR@20:{}".format(np.average(hits_20)))
    print("matrix_factorization HR@30:{}".format(np.average(hits_30)))
    print("matrix_factorization HR@40:{}".format(np.average(hits_40)))
