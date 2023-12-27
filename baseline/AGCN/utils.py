from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import numpy as np
import torch
import pandas as pd
from random import random, sample
from matplotlib import pyplot as plt
from matplotlib import rcParams
import time
import os

# 实现topn推荐
from config import candidate_num

# 解决matplotlib中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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


def get_jump_poi(user_index: int, user_item_inter: dict, item_item_inter: list, n: int):
    """
    :param user_index: 用户编号
    :param user_item_inter: 用户与poi关系
    :param item_item_inter: poi与poi关系
    :param n: 邻居跳数
    :return:
    """
    # user的高阶poi邻居
    user_item_jump_list = []
    for i in range(n):
        if i == 0:
            user_item_jump_list.append(user_item_inter[user_index])
        else:
            user_item_jump_child = []
            for j in range(len(user_item_jump_list[-1])):
                user_item_jump_child2 = []
                for k in range(len(item_item_inter)):
                    if item_item_inter[k] == user_item_jump_list[-1][j]:
                        if k % 2 == 0:
                            user_item_jump_child2.append(item_item_inter[k + 1])
                        else:
                            user_item_jump_child2.append(item_item_inter[k - 1])
                user_item_jump_child.append(user_item_jump_child2)
            user_item_jump_list.append(user_item_jump_child)
    return user_item_jump_list


def get_jump_poi2(user_index: int, user_item_inter: dict, item_item_inter: list, user_item_jump_list: list, n: int,
                  n_child: int):
    """

    """
    if n_child == 0:
        user_item_jump_list.append(user_item_inter[user_index])
        user_item_jump_list.append(
            get_jump_poi2(user_index, user_item_inter, item_item_inter, user_item_jump_list[-1], n, n_child + 1))
    else:
        if len(user_item_jump_list) == 0:
            return []
        else:
            if n != n_child:
                user_item_jump_child = []
                for j in range(len(user_item_jump_list)):
                    user_item_jump_child2 = []
                    for k in range(len(item_item_inter)):
                        if item_item_inter[k] == user_item_jump_list[j]:
                            if k % 2 == 0:
                                user_item_jump_child2.append(item_item_inter[k + 1])
                            else:
                                user_item_jump_child2.append(item_item_inter[k - 1])
                    user_item_jump_child.append(user_item_jump_child2)
                user_item_jump_list.append(user_item_jump_child)
                user_item_jump_list.append(
                    get_jump_poi2(user_index, user_item_inter, item_item_inter, user_item_jump_list[-1], n,
                                  n_child + 1))
            else:
                return user_item_jump_list
    return user_item_jump_list


def get_jump_user():
    pass


def drawer(title: str, x_label: str, y_label: str, x_data: list, y_data: list, label: str, c: list = None,
           text: list = None, is_center: bool = False):
    fig = plt.figure(title, facecolor='lightgray', figsize=(6, 6))
    ax = plt.gca()
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tick_params(labelsize=22)
    plt.scatter(x_data, y_data, s=60, c=c, cmap='brg')
    # 添加标题
    if text is not None:
        for i in range(len(x_data)):
            plt.text(x_data[i], y_data[i] + 0.1, text[i], ha="center", va="bottom", fontsize=22)

    # 将坐标远点移至图像中央位置
    if is_center:
        # plt.tick_params(axis='x', colors='red')
        # plt.tick_params(axis='y', colors='red')
        ax.spines['bottom'].set_color('red')
        ax.spines['left'].set_color('red')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.legend()
    plt.show()
    # plt.savefig(os.path.join("../images/visual", str(time.time())+".png"))
    # 避免出现数据重叠问题
    plt.clf()


def tsne(data: pd.array, n: int, learning_rate: int):
    tsner = TSNE(n_components=n, learning_rate=learning_rate)
    res = tsner.fit_transform(data)
    return res


def cluster(data: pd.array, n: int):
    model = KMeans(n_clusters=n)
    y_pred = model.fit_predict(data)
    return y_pred


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

    # print(hits_dict)
    return hits, hits_dict


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
                rank += 1 / rank
        rank_list.append(rank)
    return np.average(np.array(rank_list))


def get_dcg(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2) + 1),
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


def accuracy2(y_true, y_pred):
    """
    :param y_true:  真实标签
    :param y_pred:  预测标签
    :return: 准确率
    """
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    prediction = torch.argmax(y_pred, 1)
    correct += (prediction == y_true).sum().float()
    total += len(y_true)

    return (correct / total).cpu().detach().data.numpy()


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


def precision2(y_true, y_pred):
    label_all = y_true.cpu().detach().numpy().tolist()
    prediction = torch.argmax(y_pred, 1).cpu().detach().data.numpy().tolist()

    return precision_score(label_all, prediction)


def precision_topn(hits, N):
    """

    :param hits:
    :return: topn下的recall值
    """
    result = float(sum(hits) / (len(hits) * N))
    return result


# 计算召回率
def recall(y_true, y_pred):
    recall = 0
    for i in range(y_true.shape[0]):
        recall += recall_score(y_true[i], y_pred[i])
    recall /= y_true.shape[0]

    return recall


def recall2(y_true, y_pred):
    label_all = y_true.cpu().detach().numpy().tolist()
    prediction = torch.argmax(y_pred, 1).cpu().detach().data.numpy().tolist()

    return recall_score(label_all, prediction)


def recall_topn(hits):
    result = float(sum(hits) / len(hits))
    return result


# 计算f1得分
def f1(y_true, y_pred):
    f1 = 0
    for i in range(y_true.shape[0]):
        f1 += f1_score(y_true[i], y_pred[i])
    f1 /= y_true.shape[0]

    return f1


def f12(y_true, y_pred):
    """
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: f1-score
    """
    prob_all = []
    label_all = y_true.cpu().detach().numpy().tolist()
    prediction = torch.argmax(y_pred, 1).cpu().detach().data.numpy().tolist()

    return f1_score(label_all, prediction)


def f1_score_topn(precision_n, recall_n):
    if precision_n == 0 or recall_n == 0:
        return 0
    f1_k = (2 * precision_n * recall_n) / (precision_n + recall_n)
    return f1_k


def roc(y_true, y_pred):
    label_all = y_true.cpu().detach().numpy().tolist()
    prediction = torch.argmax(y_pred, 1).cpu().detach().data.numpy().tolist()

    return roc_curve(label_all, prediction, pos_label="1")


def get_embedding(filename):
    """
    :param filename: 文件地址
    :return: embedding
    """
    vocab, embedding = [], []
    # 打开glove.txt文件，并获得每个单词以及对应的embed
    with open(filename, 'rt', encoding='utf8') as f:
        full_content = f.read().strip().split("\n")
    # print(full_content)
    for i in range(len(full_content)):
        i_word = full_content[i].split(" ")[0]
        i_embeddings = [float(var) for var in full_content[i].split(" ")[1:]]
        vocab.append(i_word)
        embedding.append(i_embeddings)

    # 将list转化为array
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embedding)

    # 向词汇表中添加特定的新词
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')

    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)

    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))

    return embs_npa


def save_pre_words(words):
    """
    :param words: dataframe类型的句子
    :return: 保存为一个文件 保存截取等长的words 提供给embedding使用
    """
    _words = words.to_list()
    __words = []
    # print(_words)
    for i in range(len(_words)):
        __words.append(_words[i].split(" "))
    for i in range(len(__words)):
        __words[i][-1] = __words[i][-1].replace('\n', '')
    print(__words)
    return __words


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


def delete_stopwords(words):
    """
    :param words: list类型数据
    :return:
    """
    # 用于存放停用词
    _words = []
    for i in range(len(words)):
        _words.append(remove_stopwords(words[i]).split(" "))

    return _words


def delete_stopwords2(words):
    """
    :param words: list类型数据
    :return:
    """
    # 用于存放停用词
    _words = []
    for i in range(len(words)):
        _words.append(remove_stopwords(words[i]).split(" "))

    return _words


def cut_words_min(words):
    """
    :param words: 二维列表 每个列表中的每个元素表示一个单词
    :return: 截断后的words
    """
    min_len = 9999
    _words = []
    # 获取所有语句中长度最短的长度
    for i in range(len(words)):
        if len(words[i]) < min_len:
            min_len = len(words[i])
    for i in range(len(words)):
        _words.append(words[i][:min_len])

    return _words


# 根据规定长度进行截断，短的句子使用0进行补充
def cut_words_key(words, key):
    """
    :param words: 要进行截断的句子
    :param key: 规定截取的长度
    :return: 截断后的words
    """
    _words = np.asarray(words)
    # 将长度小于截取长度的句子进行0的补充
    for i in range(_words.shape[0]):
        if len(_words[i]) < key:
            index = len(_words[i])
            for j in range(key - len(_words[i])):
                _words[i].append(0)
    __words = []
    for i in range(_words.shape[0]):
        __words.append(_words[i][:key])

    return __words


def replace_char(words):
    """
    :param words: 一维列表，列表中的每一个元素是语句组成的字符串
    :return: 去掉一些标点的结果
    """
    _words = []
    for i in range(len(words)):
        _words.append(words[i].replace("!", " ").replace(".", " ").replace("?", " ").replace(",", " "))
    return _words


def charToEmbed(words, embed):
    """
    :param words: 二维列表数据
    :return: 完成embed替换的文件
    """
    embed_list = []
    word_embed = []
    zero_embed = [float(0) for i in range(50)]
    for i in range(len(words)):
        for j in range(len(words[0])):
            if words[i][j] in embed.keys():
                words[i][j] = embed[words[i][j]]
            else:
                words[i][j] = zero_embed

    return words


def get_glove(path):
    """
    :param path: txt文件
    :return: json
    """
    vocab, embedding = [], []
    # 打开glove.txt文件，并获得每个单词以及对应的embed
    with open(path, 'rt', encoding='utf8') as f:
        full_content = f.read().strip().split("\n")
    # print(full_content)
    for i in range(len(full_content)):
        i_word = full_content[i].split(" ")[0]
        i_embeddings = [float(var) for var in full_content[i].split(" ")[1:]]
        vocab.append(i_word)
        embedding.append(i_embeddings)

    # 将list转化为array
    vocab_npa = vocab
    embs_npa = embedding

    dict_words_embed = {}
    for i in range(len(vocab_npa)):
        dict_words_embed[vocab_npa[i]] = embs_npa[i]

    return dict_words_embed


# 进行随机抽样以及抽样的样本索引（每一行都是一样的索引）
def getSampleIndex1(res_p, N):
    """
    :param res_p: 表示预测的结果
    :param N: 表示随机抽取几个样本
    :return: 抽取的样本以及对应的索引值
    """
    li = [i for i in range(res_p.shape[1])]
    # 随机获取的索引值
    index = sample(li, N)
    res = torch.zeros((res_p.shape[0], res_p.shape[1]))
    for i in range(res_p.shape[0]):
        for k, v in enumerate(index):
            res[i][v] = res_p[i][v]

    return res, index


# 进行从大到小的顺序对每一行进行排序并且获得前N项的原索引
def getSampleIndex2(res_p, N):
    """
    :param res_p: 表示预测的结果
    :param N: 表示随机抽取几个样本
    :return: 排序抽取的样本以及对应的索引值
    """
    res_p = res_p.detach().numpy()
    index = []
    res = []
    for i in range(res_p.shape[0]):
        res.append(sorted(res_p[i], key=lambda x: (-x).tolist())[0:N])
        index.append(np.argsort(res_p[i]).tolist()[0:N])

    res = torch.tensor(res)
    return res, index


def save_file(filename, method="jpg"):
    pass


# 获得指标中的最大值
def max_value(value_list):
    max_ = 0
    return max_


# 绘制颜色矩阵
def draw_color_matrix(arr: list, x_title: str, y_title: str, x_dis: int, y_dis: int, save_path: str, display_ticks: bool = False):
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    if display_ticks:
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    odd_index, even_index = 1, 2
    for j in range(len(arr)):
        for i in range(len(arr[j])):
            if j % 2 == 0:
                plt.subplot(x_dis, y_dis, odd_index)
                odd_index += 2
            else:
                plt.subplot(x_dis, y_dis, even_index)
                even_index += 2
            arr2d = np.reshape(np.array(arr[j][i]), (1, -1))
            plt.imshow(arr2d, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

    plt.show()
    # plt.cla()
    # plt.savefig(save_path)

# 计算指标相对提高比

# if __name__ == '__main__':
#     txt_to_json(config.glove_path)
