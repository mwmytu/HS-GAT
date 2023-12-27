from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np
import pandas as pd
from pandas import Series
from random import random, sample
from matplotlib import pyplot as plt

# 实现topn推荐
from config import candidate_num

# class TopN:
#     def __init__(self):
#         super(TopN, self).__init__()
#
#     def getTopN(self, ratings_t, ratings_p, N, vpt):
#         """
#         :param ratings_t:   表示真实用户与事件是否交互的矩阵
#         :param ratings_p:   表示预测用于与事件是否交互的矩阵
#         :param N:   表示需要选择的用户喜好的前N项事件
#         :param vpt: 表示规定预测结果中表示用户参与事件的概率阈值
#         :return:    预测结果中用户参与事件以及
#         """
#
#         # with torch.no_grad():
#         # ratings = ratings.detach().numpy()
#         topn_list = torch.zeros((ratings.shape[0], N))
#         user_event = torch.zeros((ratings.shape[0], ratings.shape[1]))
#
#         # 获取每个用户前n个最大的事件
#         for i in range(ratings.shape[0]):
#             # 获取前n个最大的事件
#             for j in range(N):
#                 index = torch.argmax(ratings[i])
#                 # np.delete(ratings[i], index)
#                 ratings[i][index] = -1
#                 topn_list[i][j] = index
#                 user_event[i][index] = 1
#
#         # 获取topn预测结果
#         return topn_list, user_event


def dcg():
    pass


def idcg():
    pass


def ndcg():
    pass


def get_n(predict, u_i_dict):
    predict_label = np.ones((len(u_i_dict), len(u_i_dict[0]), 2), dtype=float)
    for k, v in enumerate(u_i_dict):
        for i in range(len(u_i_dict[0])):
            predict_label[v][i] = predict[v][u_i_dict[v][i]]

    return predict_label


# HR指标
def hits(u_i_dict, predict, N):
    """

    :param u_i_dict: 每个用户参与和没有参与过的项目
    :param predict: 模型预测结果
    :param N: top-N
    :return: hits
    """
    hits = []
    hits_dict = {}
    # predicted_labels = predict.detach().numpy()
    predicted_labels = predict.reshape((len(u_i_dict), candidate_num))
    # for k, (v, j) in enumerate(u_i_dict):
    for k, v in enumerate(u_i_dict):
        topn_items = []
        items_prob = []
        for i in range(len(u_i_dict[v])):
            items_prob.append(predicted_labels[v][i])
        items_index = np.argsort(np.array(items_prob))[::-1][0:N]
        # print(len(items_index))
        for i in range(len(items_index)):
            topn_items.append(u_i_dict[v][items_index[i]])
        hits_dict[v] = topn_items
        if u_i_dict[v][-1] in topn_items:
            hits.append(1)
        else:
            hits.append(0)

    return hits


# MRR指标
def mrr(hits_dict, u_i_dict):
    """

    :param hits_dict: 存储每个用户对应的预测top项目
    :param u_i_dict: 存储每个用户对应的选取的候选项目
    :return:
    """
    rank_list = []
    for k, v in enumerate(hits_dict):
        # 获取某一用户对应的topn项目
        pred_items = hits_dict[v]
        # 获取该用户喜欢的项目
        gt_items = [u_i_dict[v][-1]]
        rank = 0
        for gt_item in gt_items:
            if gt_item in pred_items:
                rank = pred_items.index(gt_item) + 1
                rank += 1/rank
        rank_list.append(rank)
    return np.average(np.array(rank_list))


def get_dcg(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
        # np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
        dtype=np.float32)


# NDCG指标
def get_ndcg(hits_dict, u_i_dict):
    # 所有用户的NDCG
    ndcg_list = []
    for k, v in enumerate(hits_dict):
        # 用户喜欢的item
        pos_items = [u_i_dict[v][-1]]
        # 模型预测该用户候选的item
        rank_list = hits_dict[v]
        relevance = np.ones_like(pos_items)
        it2rel = {it: r for it, r in zip(pos_items, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
        # print(rank_scores)
        idcg = get_dcg(relevance)

        dcg = get_dcg(rank_scores)

        # 每个用户的NDCG
        ndcg = dcg / idcg
        ndcg_list.append(ndcg)

    return np.average(np.array(ndcg_list))


# 计算准确率
def accuracy(y_true, y_pred):

    hits = []
    y_true = y_true.data.numpy()
    y_pred = y_pred.data.numpy()
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            hits.append(1)
        else:
            hits.append(0)
    # accuracy = accuracy_score(y_true, y_pred)

    return np.average(np.array(hits))


# 计算精确率
def precision(y_true, y_pred):

    pre = 0
    y_true = y_true.astype(np.int)
    y_pred = y_pred.astype(np.int)
    # y_pred = y_pred.detach().numpy()
    for i in range(y_true.shape[0]):
        pre += precision_score(y_true[i], y_pred[i])
    pre /= y_true.shape[0]
    # y_pred = y_pred.data.numpy()
    # precision = precision_score(y_true, y_pred)

    return pre


def precision_topn(hits, N):
    """

    :param hits:
    :return: topn下的recall值
    """
    if np.array(hits).sum() == 0:
        return 0
    result = float(sum(hits) / (len(hits) * N))
    return result


# 计算召回率
def recall(y_true, y_pred):

    recall = 0
    for i in range(y_true.shape[0]):
        recall += recall_score(y_true[i], y_pred[i])
    recall /= y_true.shape[0]
    
    return recall


def recall_topn(hits):

    if np.array(hits).sum() == 0:
        return 0
    result = float(sum(hits) / len(hits))
    return result


# 计算f1得分
def f1(y_true, y_pred):

    f1 = 0
    for i in range(y_true.shape[0]):
        f1 += f1_score(y_true[i], y_pred[i])
    f1 /= y_true.shape[0]

    return f1


def f1_score_topn(precision_n, recall_n):

    if precision_n == 0 or recall_n == 0:
        return 0
    f1_k = (2 * precision_n * recall_n) / (precision_n + recall_n)
    return f1_k


def save_to_path(data, path, type):
    """
    :param data: 需要保存的数据 类型为dataframe
    :param path: 需要保存的路径
    :param type: 需要保存的文件类型
    :return: null
    """
    if type == "csv":
        data.to_csv(path, encoding='utf8')
    if type == "excel":
        writer = pd.ExcelWriter(path)
        data.to_excel(writer, index=True)


def drawer(x, y, x_label, y_label, title, filename):
    """
    :param x: x轴数据
    :param y: y轴数据
    :param x_label: x轴标题
    :param y_label: y轴标题
    :param title: 图像标题
    :param filename: 文件保存名称
    :param method: 文件保存类型
    :return: 保存的图片
    """
    fig = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y)
    # plt.savefig('./images/plt' + filename)
    plt.show()


def save_file(filename, method="jpg"):

    pass


# 获得指标中的最大值
def max_value(value_list):
    max_ = 0
    return max_


def series2mean(value: Series) -> float:
    mean = value.mean()
    return mean

# if __name__ == '__main__':
#     txt_to_json(config.glove_path)
