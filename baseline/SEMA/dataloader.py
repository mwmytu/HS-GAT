import pandas as pd
import numpy as np
import json
import time
import math

from sklearn.utils import shuffle

from config import sequence_len, words_len, neighbor_num


def train_deal(dataset, train_data):
    # 组成用户-项目交互word_embed矩阵
    user_len = len(dataset["user_id"].unique())
    business_len = len(dataset["business_id"].unique())
    user_item_word = []
    item_user_word = []
    for cur_user in train_data["user_id"].unique():
        user_item_word_ = []
        user_business = train_data[train_data.user_id == cur_user]["business_id"]
        for cur_business in user_business:
            user_word = \
                train_data[(train_data.user_id == cur_user) & (train_data.business_id == cur_business)][
                    "text"].unique()[
                    0]
            user_item_word_.append(user_word)
        user_item_word.append(user_item_word_)
    for cur_business in dataset["business_id"].unique():
        item_user_word_ = []
        item_user = train_data[train_data.business_id == cur_business]["user_id"]
        for cur_user in item_user:
            item_word = train_data[(train_data.business_id == cur_business) & (train_data.user_id == cur_user)]["text"].unique()[
                0]
            item_user_word_.append(item_word)
        item_user_word.append(item_user_word_)
    # print(word_df_list)
    # 获得用户-项目交互的word
    user_item_word_list = []
    for i in range(len(user_item_word)):
        user_item_word_ = []
        if len(user_item_word[i]) >= sequence_len:
            for j in range(sequence_len):
                user_item_word_.append([int(k) for k in user_item_word[i][j].split(" ")])
        else:
            for j in range(len(user_item_word[i])):
                user_item_word_.append([int(k) for k in user_item_word[i][j].split(" ")])
            for k in range(sequence_len - len(user_item_word[i])):
                user_item_word_.append([int(0) for m in range(words_len)])
        user_item_word_list.append(user_item_word_)
    # 获得项目-用户交互的word
    item_user_word_list = []
    for i in range(len(item_user_word)):
        item_user_word_ = []
        if len(item_user_word[i]) >= sequence_len:
            for j in range(sequence_len):
                item_user_word_.append([int(k) for k in item_user_word[i][j].split(" ")])
        else:
            for j in range(len(item_user_word[i])):
                item_user_word_.append([int(k) for k in item_user_word[i][j].split(" ")])
            for k in range(sequence_len - len(item_user_word[i])):
                item_user_word_.append([int(0) for m in range(words_len)])
        item_user_word_list.append(item_user_word_)

    # 获得用户-用户交互字典
    u_u_dict = {}
    for cur_user in train_data["user_id"].unique():
        user_set = set()
        # user_list = []
        for cur_business in train_data[train_data["user_id"] == cur_user]["business_id"]:
            user_set.add(cur_business)
            # user_list.append(cur_business)
        u_u_dict[cur_user] = user_set

    # 获得事件-事件交互字典
    v_v_dict = {}
    for cur_business in dataset["business_id"].unique():
        business_set = set()
        # business_list = []
        for cur_user in train_data[train_data["business_id"] == cur_business]["user_id"]:
            # business_set.add(cur_user)
            business_set.add(cur_user)
        v_v_dict[cur_business] = business_set

    # 遍历用户-事件交互字典   将大于规定邻居数量的进行截断
    for k, v in enumerate(u_u_dict):
        if len(u_u_dict[v]) > neighbor_num:
            u_u_dict[v] = list(u_u_dict[v])[:neighbor_num]
        else:
            u_u_dict[v] = list(u_u_dict[v])
    # 遍历事件-用户交互字典   将大于规定邻居数量的进行截断
    for k, v in enumerate(v_v_dict):
        if len(v_v_dict[v]) > neighbor_num:
            v_v_dict[v] = list(v_v_dict[v])[:neighbor_num]
        else:
            v_v_dict[v] = list(v_v_dict[v])

    train_data_sort_user = train_data.sort_values(by="user_id")
    # 获得每个用户的潜在因素
    user_id_unique = train_data_sort_user["user_id"].unique().tolist()
    user_review_count_unique, user_latent_factors = [], []
    # for cur_user in user_id_unique:
    #     user_review_count_unique.append(
    #         train_data_sort_user[train_data_sort_user.user_id == cur_user]["review_count"].unique()[0])
    user_latent_factors.append(user_id_unique)
    # user_latent_factors.append(user_review_count_unique)
    # 获得每个项目的潜在因素
    dataset_sort_item = dataset.sort_values(by="business_id")
    business_id_unique = dataset_sort_item["business_id"].unique().tolist()
    business_review_count_unique, business_latent_factors = [], []
    # for cur_item in business_id_unique:
    #     business_review_count_unique.append(
    #         dataset_sort_item[dataset_sort_item.business_id == cur_item]["item_review_count"].unique()[0])
    business_latent_factors.append(business_id_unique)
    # business_latent_factors.append(business_review_count_unique)
    # 获得训练集的目标集
    target = np.zeros((user_len, business_len))
    uv_tag = []
    for k, v in enumerate(u_u_dict):
        for i in range(len(u_u_dict[v])):
            uv_tag.append(
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])][
                    "stars"].unique()[
                    0])
            target[v][u_u_dict[v][i]] = \
            train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])]["stars"].unique()[
                0]

    # print(user_item_word_list)
    return user_item_word_list, item_user_word_list, target, u_u_dict, train_data, user_latent_factors, business_latent_factors


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
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        # 根据留一法将一个参加过的事件和没参加过的事件进行拼接
        # test_items[:1]：用户id   test_items[1:]：用户参加过事件+没参加过事件
        test_items = selected_not_interacted + [i]
        u_i_dict[u] = test_items

    return u_i_dict


def get_data():
    f_bus = open("../data/yelp/yelp_academic_dataset_business.json", encoding="utf8")
    f_review = open("../data/yelp/yelp_academic_dataset_review.json", encoding="utf8")
    f_user = open("../data/yelp/yelp_academic_dataset_user.json", encoding="utf8")

    js_bus, js_review, js_user = [], [], []
    for i in range(40000):
        js_bus.append(json.loads(f_bus.readline()))
        js_review.append(json.loads(f_review.readline()))
        js_user.append(json.loads(f_user.readline()))

    business_list = ["business_id", "latitude", "longitude", "stars", "review_count"]
    review_list = ["business_id", "user_id", "text", "date", "stars"]
    user_list = ["user_id", "yelping_since", "review_count"]
    # target = ["user_id", "business_id", "stars"]

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
    # 将得分转化为百分制
    # target["stars"] = target["stars"] * 20
    # print(dataset)

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
        date = int(dataset["date"].iloc[i][11:13])
        if (date >= 7) and (date < 9):
            time_phase_list.append(3)
        elif (date >= 9) and (date < 12):
            time_phase_list.append(2)
        elif (date >= 12) and (date < 18):
            time_phase_list.append(1)
        else:
            time_phase_list.append(4)
    df_date = pd.DataFrame(time_phase_list, columns=["date"])
    dataset["date"] = df_date["date"]
    # 将用户对事件的评分进行处理，二分类（喜欢 or 不喜欢）
    dataset["stars_mean"] = dataset["stars"] / dataset["stars"].mean()
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

    dataset.sort_values(by="islike", inplace=True, ascending=True)

    # 对处理之后的数据集的id进行重新编排
    business_uni = dataset["business_id"].unique()
    user_uni = dataset["user_id"].unique()
    bus_ids_invmap = {id_: i for i, id_ in enumerate(business_uni)}
    user_ids_invmap = {id_: i for i, id_ in enumerate(user_uni)}
    dataset["business_id"].replace(bus_ids_invmap, inplace=True)
    dataset["user_id"].replace(user_ids_invmap, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # 根据评论文本获得语料库
    words_list = dataset["text"].to_list()
    word_list = []
    word_list_ = []
    for i in range(len(words_list)):
        word_list = word_list + words_list[i].split(" ")
        word_list_.append(words_list[i].split(" "))
    word_list = list(set(word_list))
    word_id_dict = {}
    for i in range(len(word_list)):
        word_id_dict[word_list[i]] = i

    # 将评论文本每个单词转化为对应的id
    word_df_list = []
    for i in range(len(word_list_)):
        word_df = pd.DataFrame(word_list_[i], columns=["text"])
        word_df.replace(word_id_dict, inplace=True)
        word_df_list.append(word_df["text"].to_list())

    # 将空值填充为0
    words_df = pd.DataFrame(word_df_list).fillna(value=0).iloc[:, :words_len]
    word_df_list = []
    for i in range(words_df.shape[0]):
        word_df_list.append(" ".join([str(int(j)) for j in words_df.iloc[i].tolist()]))
    dataset["text"] = pd.DataFrame(word_df_list)

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

    # train_data = shuffle(train_data)

    # 对business_id进行重新编号
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    return dataset, train_data, test_data


if __name__ == '__main__':
    dataset, train_data, test_data = get_data()
    user_item_word_list, item_user_word_list, uv_tag, u_u_dict, train_data = train_deal(dataset, train_data)
