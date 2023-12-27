import numpy as np
import pandas as pd
import json

import config


def train_deal(dataset, train_data):
    """

    :param dataset: 整个数据集
    :param train_data: 训练集
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

    # 遍历用户-事件交互字典   将大于规定邻居数量的进行截断
    for k, v in enumerate(u_u_dict):
        if len(u_u_dict[v]) > config.neighbor_num:
            u_u_dict[v] = list(u_u_dict[v])[:config.neighbor_num]
        else:
            u_u_dict[v] = list(u_u_dict[v])

    # 获得训练集的目标集
    target = np.zeros((user_len, business_len))
    uv_tag = []
    for k, v in enumerate(u_u_dict):
        for i in range(len(u_u_dict[v])):
            uv_tag.append(
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])]["stars"].unique()[
                    0])
            target[v][u_u_dict[v][i]] = \
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])]["stars"].unique()[
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
    user_id = train_data["user_id"].unique().tolist()
    item_id = dataset["business_id"].unique().tolist()

    return user_id, item_id, target, u_u_dict


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


def getData():
    f_bus = open("../../../data/yelp/yelp_academic_dataset_business.json", encoding="utf8")
    f_review = open("../../../data/yelp/yelp_academic_dataset_review.json", encoding="utf8")
    f_user = open("../../../data/yelp/yelp_academic_dataset_user.json", encoding="utf8")

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

    f_bus.close()
    f_review.close()

    return dataset, train_data, test_data


if __name__ == '__main__':
    dataset, train_data, test_data = getData()
    user_id, item_id, uv_tag, u_u_dict = train_deal(dataset, train_data)
