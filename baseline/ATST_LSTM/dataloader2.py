import numpy as np
import pandas as pd
import json
import time
import math
import random
from sklearn.utils import shuffle

import config


def timestamp(date):
    s_t = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    mkt = int(time.mktime(s_t))
    return mkt


def train_deal(dataset, train_data):
    """

    :param dataset: 整个数据集
    :param train_data: 训练集
    :param u_u_dict: 用户-用户交互
    :param v_v_dict: 训练集
    :return: u_v_matrix, v_u_matrix
    """
    user_len = len(dataset["user_id"].unique())
    business_len = len(dataset["business_id"].unique())

    # 获得用户-用户交互字典
    u_u_dict = {}
    for cur_user in train_data["user_id"].unique():
        user_set = set()
        # user_list = []
        for cur_business in train_data[train_data["user_id"] == cur_user]["business_id"]:
            user_set.add(cur_business)
            # user_list.append(cur_business)
        u_u_dict[cur_user] = user_set

    all_item_set = set()
    for cur_business in dataset["business_id"].unique():
        all_item_set.add(cur_business)
    u_u_dict_all = {}
    for cur_user in dataset["user_id"].unique():
        user_set = set()
        # user_list = []
        for cur_business in train_data[train_data["user_id"] == cur_user]["business_id"]:
            user_set.add(cur_business)
            # user_list.append(cur_business)
        u_u_dict_all[cur_user] = all_item_set - user_set

    # 获得事件-事件交互字典
    v_v_dict = {}
    for cur_business in dataset["business_id"].unique():
        business_set = set()
        # business_list = []
        for cur_user in train_data[train_data["business_id"] == cur_business]["user_id"]:
            # business_set.add(cur_user)
            business_set.add(cur_user)
        v_v_dict[cur_business] = business_set
    # 获得用户-事件交互矩阵
    # u_v_matrix = np.zeros((4, len(u_v_user), len(u_v_business)))
    #   1.经度矩阵
    u_v_lati_longi = np.zeros((user_len, config.neighbor_num))
    #   2.项目平均得分矩阵
    u_v_item_stars = np.zeros((user_len, config.neighbor_num))
    #   3.项目id矩阵
    u_v_vid = np.zeros((user_len, config.neighbor_num))
    #   1.2.3.4.
    u_v_star = np.zeros((user_len, config.neighbor_num))
    # 遍历用户-事件交互字典   将大于规定邻居数量的进行截断
    for k, v in enumerate(u_u_dict):
        if len(u_u_dict[v]) > config.neighbor_num:
            u_u_dict[v] = list(u_u_dict[v])[:config.neighbor_num]
        else:
            u_u_dict[v] = list(u_u_dict[v])
    # 遍历事件-用户交互字典   将大于规定邻居数量的进行截断
    for k, v in enumerate(v_v_dict):
        if len(v_v_dict[v]) > config.neighbor_num:
            v_v_dict[v] = list(v_v_dict[v])[:config.neighbor_num]
        else:
            v_v_dict[v] = list(v_v_dict[v])

    # 获取邻居的各个属性
    for k, v in enumerate(u_u_dict):
        i = 0
        for j in range(len(u_u_dict[v])):
            u_v_vid[v][j] = u_u_dict[v][j]
            u_v_lati_longi[v][j] = \
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][j])][
                    "lati_longi_aver"].unique()[0]
            u_v_item_stars[v][j] = \
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][j])][
                    "item_stars"].unique()[0]
            u_v_star[v][j] = \
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][j])][
                    "item_stars"].unique()[0]
            i += 1

    u_v_matrix = np.stack((u_v_vid, u_v_lati_longi, u_v_item_stars), axis=0)

    # 获得事件-用户交互矩阵
    # v_u_matrix = np.zeros((3, len(v_u_business), len(v_u_user)))
    #   1.评论时间矩阵
    v_u_date = np.zeros((business_len, config.neighbor_num))
    #   4.事件-用户id
    v_u_uid = np.zeros((business_len, config.neighbor_num))
    #   5.words
    words_df = pd.DataFrame(columns=["words"])
    v_u_words = np.zeros((business_len, config.neighbor_num))
    #   1.2.
    words_list = []
    for k, v in enumerate(v_v_dict):
        for j in range(len(v_v_dict[v])):
            v_u_uid[v][j] = v_v_dict[v][j]
            v_u_date[v][j] = \
                train_data[(train_data.business_id == v) & (train_data.user_id == v_v_dict[v][j])]["date"].unique()[
                    0]
            words_list.append(
                train_data[(train_data.business_id == v) & (train_data.user_id == v_v_dict[v][j])]["text"].unique()[0])

    v_u_matrix = np.stack((v_u_uid, v_u_date), axis=0)

    # 获得训练集的目标集
    target = np.zeros((user_len, business_len))
    uv_tag = []
    for k, v in enumerate(u_u_dict):
        for i in range(len(u_u_dict[v])):
            uv_tag.append(
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])]["stars"].unique()[
                    0])
            target[v][u_u_dict[v][i]] = train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])]["stars"].unique()[
                    0]

    # 构造BPR情况下的用户-喜欢-不喜欢的三元组
    # u_i_dict_3 = dict()
    # for cur_user in train_data["user_id"].unique():
    #     user_like_dislike = []
    #     user_like =train_data[(train_data.user_id == cur_user) & (train_data.islike == 1)]
    #     for i in range(user_like.shape[0]):
    #         user_dislike = train_data[(train_data.user_id == cur_user) & (train_data.islike == 0)]
    #         user_like_dislike.append(
    #             [cur_user, user_like["business_id"].iloc[i], random.choice(user_dislike["business_id"].tolist())])
    #     u_i_dict_3[cur_user] = user_like_dislike

    return u_v_matrix, v_u_matrix, target, words_list, u_u_dict, v_v_dict, u_u_dict_all, u_v_star


# 根据留一法获取用户交互的一次事件和没有交互的事件，用于推荐评测
def test_deal(dataset, test_data):
    """

    :param dataset: 整个数据集
    :param test_data:  测试集
    :return:
    """
    # 用于测试的用户-项目对
    test_user_business_set = set(zip(test_data['user_id'], test_data['business_id']))
    # 每个用户与之交互的所有条目
    user_interacted_items = dataset.groupby('user_id')['business_id'].apply(list).to_dict()
    # 所有用户交互的所有项目
    all_business = dataset["business_id"].unique()
    # 采用留一法去获得每一项仅包含该用户唯一交互的事件和剩余没有交互的事件
    # 用于保存每个用户与之交互和没有交互的事件
    u_i_dict = {}
    for (u, i) in test_user_business_set:
        # 获得某用户参加的事件
        interacted_items = user_interacted_items[u]
        # 获得某用户没有参加的事件
        not_interacted_items = set(all_business) - set(interacted_items)
        # 随机选取若干个没有参加的事件
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), config.candidate_num - 1))
        # 根据留一法将一个参加过的事件和没参加过的事件进行拼接
        # test_items[:1]：用户id   test_items[1:]：用户参加过事件+没参加过事件
        test_items = selected_not_interacted + [i]
        u_i_dict[u] = test_items

    return u_i_dict


# 获得矩阵分解的user-item评分矩阵
def get_matrix_factor():
    f_bus = open("./data/", encoding="utf8")
    f_review = open("./data/yelp_academic_dataset_review.json", encoding="utf8")
    f_user = open("./data/yelp_academic_dataset_user.json", encoding="utf8")

    js_bus, js_review, js_user = [], [], []
    for i in range(20000):
        js_bus.append(json.loads(f_bus.readline()))
        js_review.append(json.loads(f_review.readline()))
        js_user.append(json.loads(f_user.readline()))

    business_list = ["business_id", "latitude", "longitude", "stars", "review_count"]
    review_list = ["business_id", "user_id", "text", "date", "stars"]
    user_list = ["user_id", "yelping_since", "review_count"]
    # 特征
    df_bus = pd.DataFrame(js_bus)
    df_review = pd.DataFrame(js_review)
    df_user = pd.DataFrame(js_user)
    fea_bus = df_bus[business_list]
    fea_review = df_review[review_list]
    fea_user = df_user[user_list]

    bus_rename = ["business_id", "latitude", "longitude", "item_stars", "item_review_count"]
    fea_bus.columns = bus_rename

    # 特征集
    fea_user_review = pd.merge(fea_user, fea_review, on=["user_id"])
    # print(fea_user_review)
    dataset = pd.merge(fea_bus, fea_user_review, on=["business_id"])

    for cur_user in dataset["user_id"].unique().tolist():
        user_item = dataset[(dataset.user_id == cur_user)]["business_id"].tolist()
        if len(user_item) < 2:
            index_list = dataset[dataset.user_id == cur_user].index.tolist()
            dataset.drop(index_list, inplace=True)

    # 对处理之后的数据集的id进行重新编排
    business_uni = dataset["business_id"].unique()
    user_uni = dataset["user_id"].unique()
    bus_ids_invmap = {id_: i for i, id_ in enumerate(business_uni)}
    user_ids_invmap = {id_: i for i, id_ in enumerate(user_uni)}
    dataset["business_id"].replace(bus_ids_invmap, inplace=True)
    dataset["user_id"].replace(user_ids_invmap, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    users = dataset['user_id'].unique()
    dataset_columns = dataset.columns.tolist()
    # 训练集
    train_data = pd.DataFrame(columns=dataset_columns)
    for i in range(dataset.shape[1]):
        train_data.iloc[:, i] = dataset.iloc[:, i]
    # 测试集
    test_data = pd.DataFrame(columns=dataset_columns)
    for i in range(len(users)):
        user_stars = dataset[(dataset.user_id == users[i])]
        user_test_data = user_stars.iloc[-1:]
        train_data.drop(index=[user_test_data.index[0]], inplace=True)
        test_data = test_data.append(user_test_data, ignore_index=True)

    # 对business_id进行重新编号
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    user_len = len(dataset["user_id"].unique())
    business_len = len(dataset["business_id"].unique())
    # 获得训练集中的用于矩阵分解的用户-项目得分矩阵
    user_item_matrix = np.zeros((user_len, business_len))
    for cur_user in train_data["user_id"].unique().tolist():
        for cur_business in train_data[train_data.user_id == cur_user]["business_id"].unique().tolist():
            user_item_matrix[cur_user][cur_business] = \
                train_data[(train_data.user_id == cur_user) & (train_data.business_id == cur_business)][
                    "stars"].unique()[0]

    return dataset, user_item_matrix, test_data


def getData():
    list_columns = ["id", "latitude", "longitude", "review_scores_rating"]
    review_columns = ["listing_id", "reviewer_id", "date", "comments"]
    f_list = pd.read_csv("../data/Boston/listings.csv", encoding="utf8")[list_columns]
    f_review = pd.read_csv("../data/Boston/reviews.csv", encoding="utf8")[review_columns]
    f_list.columns = ["business_id", "latitude", "longitude", "stars"]
    f_review.columns = ["business_id", "user_id", "date", "text"]
    dataset = pd.merge(f_list, f_review, on=["business_id"]).iloc[:35000, :]

    # ======数据处理区域开始====== #
    # 将数据集中个数小于规定个数的用户和事件删除
    users = dataset['user_id'].unique()
    items = dataset['business_id'].unique()
    # pd.value_counts()：对pd中series里面的数值进行归类并统计每一类别出现的次数并进行排序
    item_count = dataset['business_id'].value_counts()
    item_count.name = 'item_count'
    # 将用户交互项目的出现次数合并到ratings中
    dataset = dataset.join(item_count, on='business_id')
    # 将与项目交互用户的出现次数合并到ratings中
    user_count = dataset['user_id'].value_counts()
    user_count.name = 'user_count'
    dataset = dataset.join(user_count, on='user_id')
    # 将交互次数比较少的用户或着项目进行删除
    dataset = dataset[(dataset['user_count'] >= 3) & (dataset['item_count'] >= 3)]
    users = dataset['user_id'].unique()
    items = dataset['business_id'].unique()
    # del删除  释放内存
    del dataset['user_count']
    del dataset['item_count']
    # 重置索引
    dataset.reset_index(drop=True, inplace=True)

    # 对数据集进行id替换
    business_uni = dataset["business_id"].unique()
    user_uni = dataset["user_id"].unique()
    bus_ids_invmap = {id_: i for i, id_ in enumerate(business_uni)}
    user_ids_invmap = {id_: i for i, id_ in enumerate(user_uni)}
    dataset["business_id"].replace(bus_ids_invmap, inplace=True)
    dataset["user_id"].replace(user_ids_invmap, inplace=True)

    # 对数据集的字段进行处理
    # 人类按照心情变化，一天可以分为4个阶段
    #   第一个阶段：7-9   第二个阶段：9-12  第三个阶段：12-18 第四个阶段：18-7
    time_phase_list = []
    for i in range(dataset.shape[0]):
        date = int(dataset["date"].iloc[i][5:7])
        if (date >= 3) and (date <= 5):
            time_phase_list.append(3)
        elif (date >= 6) and (date <= 8):
            time_phase_list.append(2)
        elif (date >= 9) and (date <= 12):
            time_phase_list.append(1)
        else:
            time_phase_list.append(4)
    df_date = pd.DataFrame(time_phase_list, columns=["date"])
    dataset["date"] = df_date["date"]
    # 获得每个事件的平均得分
    business_mean_score = []
    business_mean_score_dict = {}
    for cur_business in dataset["business_id"].unique():
        bus_len = len(dataset[dataset.business_id == cur_business])
        bus_score_sum = dataset[dataset.business_id == cur_business]["stars"].sum()
        bus_score_mean = bus_score_sum / bus_len
        business_mean_score_dict[cur_business] = bus_score_mean
    for cur_business in dataset["business_id"]:
        business_mean_score.append(int(business_mean_score_dict[cur_business]*20))
    dataset["item_stars"] = pd.DataFrame(business_mean_score)

    # 将用户对事件的评分进行处理，二分类（喜欢 or 不喜欢）
    # dataset["stars_mean"] = dataset["stars"] / dataset["stars"].mean()
    like_list = []
    for i in range(dataset.shape[0]):
        if dataset.iloc[i]["stars"] >= 4:
            like_list.append(1)
        else:
            like_list.append(0)
    dataset["islike"] = pd.DataFrame(like_list)
    # del dataset["stars"]
    # 对数据集中的经纬度采用中心法处理
    # 获取到每个用户参与事件的经纬度的平均经纬度
    users = dataset['user_id'].unique()
    items = dataset['business_id'].unique()
    user_lati_longi = {}
    for i in range(len(users)):
        user_latitude = dataset[dataset.user_id == users[i]]["latitude"]
        user_longitude = dataset[dataset.user_id == users[i]]["longitude"]
        lati_aver = user_latitude.sum() / len(user_latitude)
        longi_aver = user_longitude.sum() / len(user_longitude)
        user_lati_longi[users[i]] = [lati_aver, longi_aver]

    user_lati_longi_aver_list = []
    for i in range(dataset.shape[0]):
        user_latitude = dataset.iloc[i]["latitude"]
        user_longitude = dataset.iloc[i]["longitude"]
        # 获得距离中心点的距离
        user_lati_longi_aver = math.sqrt((user_latitude - user_lati_longi[dataset.iloc[i]["user_id"]][0]) ** 2 + (
                user_longitude - user_lati_longi[dataset.iloc[i]["user_id"]][1]) ** 2)
        user_lati_longi_aver_list.append(user_lati_longi_aver)
    user_lati_longi_aver_df = pd.DataFrame(user_lati_longi_aver_list, columns=["lati_longi_aver"])
    dataset["lati_longi_aver"] = user_lati_longi_aver_df["lati_longi_aver"]
    del dataset["latitude"]
    del dataset["longitude"]

    for i in range(len(users)):
        user_islike = dataset[(dataset.user_id == users[i])]["islike"]
        if user_islike.sum() < 2:
            index_list = dataset[dataset.user_id == users[i]].index.tolist()
            dataset.drop(index_list, inplace=True)

    # 对处理之后的数据集的id进行重新编排
    business_uni = dataset["business_id"].unique()
    user_uni = dataset["user_id"].unique()
    bus_ids_invmap = {id_: i for i, id_ in enumerate(business_uni)}
    user_ids_invmap = {id_: i for i, id_ in enumerate(user_uni)}
    dataset["business_id"].replace(bus_ids_invmap, inplace=True)
    dataset["user_id"].replace(user_ids_invmap, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # 采用留一法获取训练集以及测试集
    # 删除喜欢事件个数小于2的用户
    users = dataset['user_id'].unique()
    dataset_columns = dataset.columns.tolist()
    # 训练集
    train_data = pd.DataFrame(columns=dataset_columns)
    for i in range(dataset.shape[1]):
        train_data.iloc[:, i] = dataset.iloc[:, i]
    # 测试集
    test_data = pd.DataFrame(columns=dataset_columns)
    for i in range(len(users)):
        user_stars = dataset[(dataset.user_id == users[i]) & dataset.islike == 1]
        user_test_data = user_stars.iloc[-1:]
        train_data.drop(index=[user_test_data.index[0]], inplace=True)
        test_data = test_data.append(user_test_data, ignore_index=True)

    # 对business_id进行重新编号
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    # target["islike"] = pd.DataFrame(like_list)

    # 通过数据集获取用户-用户邻接矩阵 事件-事件邻接矩阵
    # 每个用户参与的事件
    u_u_dict = {}
    ug_u_u = []
    ug_u_u_reverse = []
    for cur_user in dataset["user_id"].unique():
        user_set = set()
        # user_list = []
        for cur_business in train_data[train_data["user_id"] == cur_user]["business_id"]:
            user_set.add(cur_business)
            # user_list.append(cur_business)
        u_u_dict[cur_user] = user_set
        # u_u_dict[cur_user] = user_list
    key_list = []
    for k in u_u_dict.keys():
        key_list.append(k)
    for k, v in enumerate(u_u_dict):
        if k <= len(u_u_dict) - 2:
            for k2, v2 in enumerate(key_list[k + 1:]):
                # 两个用户有相同的交互事件
                if u_u_dict[v].intersection(u_u_dict[v2]):
                    ug_u_u.append(v)
                    ug_u_u.append(v2)
                    ug_u_u_reverse.append(v2)
                    ug_u_u_reverse.append(v)
    # 事件和事件的邻接矩阵
    # 参与每个事件的用户
    v_v_dict = {}
    ug_v_v = []
    ug_v_v_reverse = []
    for cur_business in dataset["business_id"].unique():
        # business_set = set()
        business_list = []
        for cur_user in train_data[train_data["business_id"] == cur_business]["user_id"]:
            # business_set.add(cur_user)
            business_list.append(cur_user)
        v_v_dict[cur_business] = business_list
    key_list = []
    for k in v_v_dict.keys():
        key_list.append(k)
    for k, v in enumerate(v_v_dict):
        if k <= len(v_v_dict) - 2:
            for k2, v2 in enumerate(key_list[k + 1:]):
                if set(v_v_dict[v]).intersection(set(v_v_dict[v2])):
                    ug_v_v.append(v)
                    ug_v_v.append(v2)
                    ug_v_v_reverse.append(v2)
                    ug_v_v_reverse.append(v)

    # 获得符合框架形式的邻接阵
    ug_u_u2 = []
    ug_v_v2 = []
    ug_u_u2.append(ug_u_u)
    ug_u_u2.append(ug_u_u_reverse)
    ug_v_v2.append(ug_v_v)
    ug_v_v2.append(ug_v_v_reverse)

    return dataset, train_data, test_data, ug_u_u2, ug_v_v2
    # 将date转化为时间戳的形式
    # date_list = v_u_fea["date"].to_list()
    # since_list = v_u_fea["yelping_since"].to_list()
    # datastamp_list, sincestamp_list = [], []
    # for i in range(len(date_list)):
    #     datastamp_list.append(timestamp(date_list[i]))
    #     sincestamp_list.append(timestamp(since_list[i]))
    # print(datastamp_list)
    # stamp_arr = np.stack((np.asarray(datastamp_list), np.asarray(sincestamp_list)), axis=-1)
    # datastamp = pd.DataFrame(stamp_arr, columns=["timestamp", "sincestamp"])
    # v_u_fea["timestamp"] = datastamp["timestamp"]
    # v_u_fea["sincestamp"] = datastamp["sincestamp"]

    # # 进行id的替换
    # uv_bus_ids_invmap = {id_: i for i, id_ in enumerate(u_v_business)}
    # uv_user_ids_invmap = {id_: i for i, id_ in enumerate(u_v_user)}
    # u_v_fea["business_id"].replace(uv_bus_ids_invmap, inplace=True)
    # u_v_fea["user_id"].replace(uv_user_ids_invmap, inplace=True)
    # vu_bus_ids_invmap = {id_: i for i, id_ in enumerate(v_u_business)}
    # vu_user_ids_invmap = {id_: i for i, id_ in enumerate(v_u_user)}
    # v_u_fea["business_id"].replace(vu_bus_ids_invmap, inplace=True)
    # v_u_fea["user_id"].replace(vu_user_ids_invmap, inplace=True)
    # tag_bus_ids_invmap = {id_: i for i, id_ in enumerate(tag_business)}
    # tag_user_ids_invmap = {id_: i for i, id_ in enumerate(tag_user)}
    # target["business_id"].replace(tag_bus_ids_invmap, inplace=True)
    # target["user_id"].replace(tag_user_ids_invmap, inplace=True)

    # 根据用户对事件的评分规定用户对事件是否喜欢
    # user_stars_dict = {}
    # for cur_user in target["user_id"].unique():
    #     user_stars_dict[cur_user] = [target[target.user_id == cur_user]["stars"].min(),
    #                                  target[target.user_id == cur_user]["stars"].max()]
    # like_list = []
    # for i in range(target.shape[0]):
    #     cur_user = target.loc[i]["user_id"]
    #     if user_stars_dict[cur_user][0] == user_stars_dict[cur_user][1]:
    #         like_list.append(1)
    #     else:
    #         like_list.append((target.loc[i]["stars"] - user_stars_dict[cur_user][0]) / (
    #                 user_stars_dict[cur_user][1] - user_stars_dict[cur_user][0]))
    # like_df = pd.DataFrame(like_list, columns=["islike"])
    # target["islike"] = like_df

    # print(v_u_date)
    # print(v_u_yelping_since)

    # print(u_v_fea)
    # print(v_u_fea)
    # return u_v_matrix, v_u_matrix, v_u_fea, uv_tag


if __name__ == '__main__':
    dataset, train_data, test_data, ug_u_u2, ug_v_v2 = getData()
    u_v_matrix, v_u_matrix, uv_tag, words_list, u_u_dict, v_v_dict = train_deal(dataset,
                                                                                train_data)
    # u_i_dict = test_deal(dataset, test_data)

    # l = 0
    # for k, v in enumerate(u_i_dict):
    #     for i in range(len(u_i_dict[v])):
    #         l += 1
    # print(l)
