import pandas as pd
import numpy as np
import json
import time
import math

from sklearn.utils import shuffle

from config import sequence_len, words_len, neighbor_num


def train_deal(dataset, train_data):
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

    # 获得训练集的目标集
    uv_tag = []
    for k, v in enumerate(u_u_dict):
        for i in range(len(u_u_dict[v])):
            uv_tag.append(
                train_data[(train_data.user_id == v) & (train_data.business_id == u_u_dict[v][i])][
                    "stars"].unique()[
                    0])

    # print(user_item_word_list)
    return uv_tag, u_u_dict, train_data, u_u_dict_all


# 获得矩阵分解的user-item评分矩阵
def get_matrix_factor(dataset, train_data):
    user_len = len(dataset["user_id"].unique())
    business_len = len(dataset["business_id"].unique())
    # 获得训练集中的用于矩阵分解的用户-项目得分矩阵
    user_item_matrix = np.zeros((user_len, business_len))
    for cur_user in train_data["user_id"].unique().tolist():
        for cur_business in train_data[train_data.user_id == cur_user]["business_id"].unique().tolist():
            user_item_matrix[cur_user][cur_business] = \
                train_data[(train_data.user_id == cur_user) & (train_data.business_id == cur_business)][
                    "stars"].unique()[0]

    return user_item_matrix


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
    list_columns = ["id", "latitude", "longitude", "review_scores_rating"]
    review_columns = ["listing_id", "reviewer_id", "date", "comments"]
    f_list = pd.read_csv("../../data/Boston/listings.csv", encoding="utf8")[list_columns]
    f_review = pd.read_csv("../../data/Boston/reviews.csv", encoding="utf8")[review_columns]
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
        business_mean_score.append(int(business_mean_score_dict[cur_business] * 20))
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

    dataset.sort_values(by="islike", inplace=True, ascending=True)

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

    # train_data = shuffle(train_data)

    # 对business_id进行重新编号
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

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
    user_num = len(dataset["user_id"].unique())
    item_num = len(dataset["business_id"].unique())
    u_u_arr2d = np.zeros((user_num, user_num), dtype=np.int32)
    for k, v in enumerate(u_u_dict):
        if k <= len(u_u_dict) - 2:
            for k2, v2 in enumerate(key_list[k + 1:]):
                # 两个用户有相同的交互事件
                if u_u_dict[v].intersection(u_u_dict[v2]):
                    ug_u_u.append(v)
                    ug_u_u.append(v2)
                    ug_u_u_reverse.append(v2)
                    ug_u_u_reverse.append(v)
                    # 用户-用户交互矩阵
                    u_u_arr2d[v][v2] = 1
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
    v_v_arr2d = np.zeros((item_num, item_num), dtype=np.int32)
    for k, v in enumerate(v_v_dict):
        if k <= len(v_v_dict) - 2:
            for k2, v2 in enumerate(key_list[k + 1:]):
                if set(v_v_dict[v]).intersection(set(v_v_dict[v2])):
                    ug_v_v.append(v)
                    ug_v_v.append(v2)
                    ug_v_v_reverse.append(v2)
                    ug_v_v_reverse.append(v)
                    # 项目-项目交互矩阵
                    v_v_arr2d[v][v2] = 1

    # 获得符合框架形式的邻接阵
    ug_u_u2 = []
    ug_v_v2 = []
    ug_u_u2.append(ug_u_u)
    ug_u_u2.append(ug_u_u_reverse)
    ug_v_v2.append(ug_v_v)
    ug_v_v2.append(ug_v_v_reverse)

    # 用户-项目交互矩阵
    #   1.所有用户参与各个项目的总次数
    u_v_arr2d = np.zeros((item_num, 1), dtype=np.int32)
    for cur_business in dataset["business_id"].unique():
        u_v_arr2d[cur_business][0] = len(train_data[train_data.business_id == cur_business])
    #   2.获得每个用户在好友下参与各个项目的次数
    u_v_arr2d_ = np.zeros((user_num, item_num), dtype=np.int32)
    for cur_user in train_data["user_id"].unique():
        for cur_business in train_data[train_data.user_id == cur_user]["business_id"]:
            u_v_arr2d_[cur_user][cur_business] = u_v_arr2d[cur_business][0]

    return dataset, train_data, test_data, u_v_arr2d_, ug_u_u2, ug_v_v2


if __name__ == '__main__':
    dataset, train_data, test_data, u_v_arr2d_, ug_u_u2, ug_v_v2 = get_data()
    # user_item_word_list, item_user_word_list, uv_tag, u_u_dict, train_data = train_deal(dataset, train_data)
