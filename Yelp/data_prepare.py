import numpy as np
import pandas as pd
import torch
import os
import config


# 获得lable集
class Lable():
    def __init__(self):
        super(Lable, self).__init__()

    # 获取每个用户参与事件总个数
    def getLableSum(self, ratings):
        users = ratings['user_id'].unique()
        # events = ratings['event_id'].unique()

        events_num = np.zeros(len(users))

        for cur_user in users:
            for i in enumerate(ratings[ratings.user_id == cur_user]["event_id"]):
                events_num[cur_user] += 1

        return events_num

    def getLableNum(self, ratings):

        # 获取用户参与每个事件的个数
        users = ratings['user_id'].unique()
        events = ratings['event_id'].unique()

        user_event_num = np.zeros((len(users), len(events)))

        for cur_user in users:
            events_uni = ratings[ratings.user_id == cur_user]["event_id"].unique()
            events_ = ratings[ratings.user_id == cur_user]["event_id"]
            for k1, v1 in enumerate(events_uni):
                for k2, v2 in enumerate(events_):
                    if v2 == v1:
                        user_event_num[cur_user][v1] += 1

        return user_event_num

    def getNum(self, ratings):

        users = ratings['user_id'].unique()
        events = ratings['event_id'].unique()

        events_num = self.getLableSum(ratings)
        user_event_num = self.getLableNum(ratings)
        user_event_prob = np.zeros((len(users), len(events)))

        # 获取每个用户参与各个事件的概率 作为lable
        for cur_user in users:
            user_event_num_ = user_event_num[cur_user]
            for k, v in enumerate(user_event_num_):
                user_event_prob[cur_user][k] = v / events_num[cur_user]

        return user_event_prob.astype(np.float64)

    def getLable(self, ratings, index_list):

        ratings = ratings.detach().numpy()
        index_list = index_list.numpy()
        index_list = index_list.astype(np.int)
        lable = np.zeros((ratings.shape[0], ratings.shape[1]))

        for i in range(ratings.shape[0]):
            for j in range(index_list.shape[1]):
                lable[i][index_list[i][j]] = ratings[i][index_list[i][j]]

        return lable

    # 获取用户与事件是否进行交互的矩阵 有为1，没有为0
    def getLable_(self, ratings):
        """
        :param ratings: 真实数据集
        :return: 用户与事件发生交互的矩阵
        """
        users = ratings['user_id'].unique()
        events = ratings['event_id'].unique()

        label = np.zeros((len(users), len(events)))

        for cur_user in users:
            events_uni = ratings[ratings.user_id == cur_user]["event_id"].unique()
            for i in range(label.shape[0]):
                for k1, v1 in enumerate(events_uni):
                    label[cur_user][v1] = 1

        return label

# 将各个数据转化为对应的embed
def init_embed(ratings):

    users = ratings['user_id'].unique()
    # print(users.index)
    events = ratings['event_id'].unique()
    week = ratings["week"].unique()
    dis = ratings["dis"].unique()

    len_users = len(users)
    len_events = len(events)
    len_week = len(week)
    len_dis = len(dis)

    user_embed_matrix = np.zeros((len_users, config.n_emb))
    event_embed_matrix = np.zeros((len_events, config.n_emb))
    week_embed_matrix = np.zeros((len_week, config.n_emb))
    dis_embed_matrix = np.zeros((len_dis, config.n_emb))
    for i in range(len_users):
        user_embed_matrix[i] = torch.normal(0, 1, size=(1, 50))
    for i in range(len_events):
        event_embed_matrix[i] = torch.normal(0, 1, size=(1, 50))
    for i in range(len_week):
        week_embed_matrix[i] = torch.normal(0, 1, size=(1, 50))
    for i in range(len_dis):
        dis_embed_matrix[i] = torch.normal(0, 1, size=(1, 50))

    print(user_embed_matrix, user_embed_matrix.shape)

    return pd.DataFrame(user_embed_matrix), pd.DataFrame(event_embed_matrix), pd.DataFrame(week_embed_matrix), pd.DataFrame(dis_embed_matrix)

def getData(ratings):

    users = ratings['user_id'].unique()
    events = ratings['event_id'].unique()

    full_data = []

    max_len_user = 0
    for cur_user in users:
        if len(ratings[ratings.user_id == cur_user].index) > max_len_user:
            max_len_user = len(ratings[ratings.user_id == cur_user]["event_id"].unique())

    max_len_event = 0
    for cur_event in events:
        if len(ratings[ratings.event_id == cur_event].index) > max_len_event:
            max_len_event = len(ratings[ratings.event_id == cur_event]["user_id"].unique())

    print(max_len_user)
    print(max_len_event)

    # 获得用户数和项目数
    num_users = len(users)
    num_events = len(events)

    # 获得用户的邻居矩阵
    adj_user = np.zeros((len(users), num_events))
    adj_week_user_event = np.zeros((len(users), num_events))
    adj_dis_user_event = np.zeros((len(users), num_events))
    adj_time_user_event = np.zeros((len(users), num_events))
    # 获得事件的邻居矩阵
    adj_event = np.zeros((len(events), num_users))
    adj_week_event_user = np.zeros((len(events), num_users))
    adj_dis_event_user = np.zeros((len(events), num_users))
    adj_time_event_user = np.zeros((len(events), num_users))
    # adj_content_user_event = np.zeros((len(users), max_len))
    # adj_content_event_user = np.zeros((len(users), max_len))

    # 获取与用户发生交互的各个矩阵
    for cur_user in users:

        # 获得用户-项目是否交互的矩阵
        for index in ratings[ratings.user_id == cur_user]["event_id"].values:
            adj_user[cur_user][index] = 1

        # 获得用户项目交互次数矩阵
        for index in ratings[ratings.user_id == cur_user]["event_id"].values:
            adj_week_user_event[cur_user][index] += 1

        # 获得用户项目距离平均值矩阵
        for index in ratings[ratings.user_id == cur_user]["event_id"].unique():
            # 交互的次数
            num = 0
            # 交互的总距离
            dis_sum = 0
            # 索引
            k = 0
            for i, j in zip(ratings[ratings.user_id == cur_user]["event_id"].values, ratings[ratings.user_id == cur_user]["event_id"].index):
                if i == index:
                    dis_sum = dis_sum + ratings[ratings.user_id == cur_user]["dis"][j]
                    num += 1
                k += 1
                    # dis_num = dis_num + ratings[]
            # 获取与用户发生交互项目的次数
            # adj_week_user_event[cur_user][] = week_count
            # 获取与用户发生交互项目的位置
            adj_dis_user_event[cur_user][index] = dis_sum / num

    # 获取与项目发生交互的各个矩阵
    for cur_event in events:

        # 获得项目-用户是否交互的矩阵
        for index in ratings[ratings.event_id == cur_event]["user_id"].values:
            adj_event[cur_event][index] = 1

        # 获得用户项目交互次数矩阵
        for index in ratings[ratings.event_id == cur_event]["user_id"].values:
            adj_week_event_user[cur_event][index] += 1

        # 获得用户项目距离平均值矩阵
        for index in ratings[ratings.event_id == cur_event]["user_id"].unique():
            # 交互的次数
            num = 0
            # 交互的总距离
            dis_sum = 0
            # 索引
            k = 0
            for i, j in zip(ratings[ratings.event_id == cur_event]["user_id"].values,
                            ratings[ratings.event_id == cur_event]["user_id"].index):
                if i == index:
                    dis_sum = dis_sum + ratings[ratings.event_id == cur_event]["dis"][j]
                    num += 1
                k += 1
                # dis_num = dis_num + ratings[]
            # 获取与用户发生交互项目的次数
            # adj_week_user_event[cur_user][] = week_count
            # 获取与用户发生交互项目的位置
            adj_dis_event_user[cur_event][index] = dis_sum / num

    # 获取用户与事件之间的时间权重
    for cur_user in users:
        # 获取用户参与的事件
        events = ratings[ratings.user_id == cur_user]["event_id"].unique()
        for cur_event in events:
            # 获得用户参与时间的最小值以及最大值
            min_week = ratings[(ratings.user_id == cur_user) & (ratings.event_id == cur_event)]["week"].min()
            max_week = ratings[(ratings.user_id == cur_user) & (ratings.event_id == cur_event)]["week"].max()
            # 获得用户参与事件的时间最小值和最大值的比值
            # 根据比值判断用户一周中参与事件的时间间隔比率
            prob = max_week / min_week
            adj_time_user_event[cur_user][cur_event] = prob

    # 获取事件与用户之间的时间权重
    for cur_event in events:
        # 获取用户参与的事件
        users = ratings[ratings.event_id == cur_event]["user_id"].unique()
        for cur_user in users:
            # 获得用户参与时间的最小值以及最大值
            min_week = ratings[(ratings.event_id == cur_event) & (ratings.user_id == cur_user)]["week"].min()
            max_week = ratings[(ratings.event_id == cur_event) & (ratings.user_id == cur_user)]["week"].max()
            # 获得用户参与事件的时间最小值和最大值的比值
            # 根据比值判断用户一周中参与事件的时间间隔比率
            prob = max_week / min_week
            adj_time_event_user[cur_event][cur_user] = prob

    return adj_user, adj_week_user_event, adj_dis_user_event, adj_event, adj_week_event_user, adj_dis_event_user, adj_time_user_event, adj_time_event_user

    # print(adj_user.shape, adj_week_user_event.shape, adj_dis_user_event.shape)

    # for cur_event in events:
    #     # print(cur_user)
    #     for j in range(len(ratings[ratings.event_id == cur_event].index) - 2):
    #         # print(ratings[ratings.user_id == cur_user]["text"][j:j+1])
    #         adj_event[cur_event][j] = ratings[ratings.event_id == cur_event]["user_id"][j:j + 1]
    #         adj_week_event_user[cur_event][j] = ratings[ratings.event_id == cur_event]["week"][j:j + 1]
    #         adj_dis_event_user[cur_event][j] = ratings[ratings.event_id == cur_event]["dis"][j:j + 1]

    # print(adj_event.shape, adj_week_event_user.shape, adj_dis_event_user.shape)
    # print(adj_user.shape)

    # 存储每个项目与各个用户交互的项目介绍
    # adj_content_user_event = {cur_user: ratings[ratings.user_id == cur_user]["text"].tolist() for cur_user in users}

    # 存储每个项目与各个用户交互的项目介绍
    # adj_content_event_user = {cur_user: ratings[ratings.user_id == cur_user]["text"].tolist() for cur_user in users}

    # return adj_user, adj_event, adj_week_user_event, adj_week_event_user, adj_dis_user_event, adj_dis_event_user

# 数据准备
def data_partition(fname):
    # 存储获取到的数据集中的数据
    ratings = []
    with open(os.path.join(fname, config.data_filename)) as f:
        # 遍历文件中的每一行
        for l in f:
            l = l.split(",")
            user_id = int(l[0])
            event_id = int(l[1])
            week = int(l[2])
            dis = float(l[3])
            text = l[4]
            ratings.append({
                'user_id': user_id,
                'event_id': event_id,
                'week': week,
                'dis': dis,
                'text': text
            })
    # 将获取到的数据转化为DataFrame对象
    ratings = pd.DataFrame(ratings)
    ratings = ratings[:500]

    users = ratings['user_id'].unique()
    events = ratings['event_id'].unique()

    # 进行数据清洗操作
    # for i in range(1000):
    #     # pd.value_counts()：对pd中series里面的数值进行归类并统计每一类别出现的次数并进行排序
    #     event_count = ratings['event_id'].value_counts()
    #     event_count.name = 'event_count'
    #     # 将用户交互项目的出现次数合并到ratings中
    #     ratings = ratings.join(event_count, on='event_id')
    #     # 将与项目交互用户的出现次数合并到ratings中
    #     user_count = ratings['user_id'].value_counts()
    #     user_count.name = 'user_count'
    #     ratings = ratings.join(user_count, on='user_id')
    #     # 将交互次数比较少的用户或着项目进行删除
    #     ratings = ratings[(ratings['user_count'] >= 3) & (ratings['event_count'] >= 3)]
    #     # 如果数据中不存在每一用户数量和每一项目数量低于规定数量的情况 则退出循环
    #     if len(ratings['user_id'].unique()) == len(users) and len(ratings['event_id'].unique()) == len(events):
    #         break
    #     users = ratings['user_id'].unique()
    #     events = ratings['event_id'].unique()
    #     # del删除  释放内存
    #     del ratings['user_count']
    #     del ratings['event_count']
    # # 删除添加的user_count与item_count字段
    # del ratings['user_count']
    # del ratings['event_count']

    # print(ratings[:3000])
    #
    # # 获得交互时间与最早交互时间的时间差 用户排序
    # ratings['timestamp'] = ratings['timestamp'] - min(ratings['timestamp'])
    users = ratings['user_id'].unique()
    events = ratings['event_id'].unique()

    # 将用户和id与映射的索引保存在字典中
    user_ids_invmap = {id_: i for i, id_ in enumerate(users)}
    event_ids_invmap = {id_: i for i, id_ in enumerate(events)}
    # 将用户和项目id替换为索引
    ratings['user_id'].replace(user_ids_invmap, inplace=True)
    ratings['event_id'].replace(event_ids_invmap, inplace=True)

    # user_embed, event_embed, week_embed, dis_embed = init_embed(ratings)
    #
    # user_embed = [user_embed.loc[:, :]]
    # print(len(ratings))

    # ratings = shuffle(ratings)

    len_ratings = len(ratings["user_id"])
    # 划分训练集、测试集、验证集
    trian_ratings = ratings[:int(len_ratings * 0.8)]
    # value_ratings = ratings[int(len_ratings * 0.8): int(len_ratings * 0.9)]
    test_ratings = ratings[int(len_ratings * 0.8):]


    # 获取训练集中的数据
    tr_adj_user, tr_adj_week_user_event, tr_adj_dis_user_event, tr_adj_event, tr_adj_week_event_user, tr_adj_dis_event_user, tr_adj_time_user_event, tr_adj_time_event_user = getData(trian_ratings)

    # 对次数和距离进行归一化
    # tr_adj_user = normalize(tr_adj_user)
    # tr_adj_event = normalize(tr_adj_event)
    # tr_adj_week_user_event = normalize(tr_adj_week_user_event)
    # tr_adj_dis_user_event = normalize(tr_adj_dis_user_event)
    # tr_adj_week_event_user = normalize(tr_adj_week_event_user)
    # tr_adj_dis_event_user = normalize(tr_adj_dis_event_user)

    tr_adj_time_week_user_event = np.zeros((tr_adj_time_user_event.shape[0], tr_adj_time_user_event.shape[1]))
    for i in range(tr_adj_time_user_event.shape[0]):
        for j in range(tr_adj_time_user_event.shape[1]):
            tr_adj_time_week_user_event[i][j] = tr_adj_time_user_event[i][j] * tr_adj_week_user_event[i][j]

    tr_adj_time_week_event_user = np.zeros((tr_adj_time_event_user.shape[0], tr_adj_time_event_user.shape[1]))
    for i in range(tr_adj_time_event_user.shape[0]):
        for j in range(tr_adj_time_event_user.shape[1]):
            tr_adj_time_week_event_user[i][j] = tr_adj_time_event_user[i][j] * tr_adj_week_event_user[i][j]

    # tr_adj_time_week_user_event = normalize(tr_adj_time_week_user_event)
    # tr_adj_time_week_event_user = normalize(tr_adj_time_week_event_user)

    # tr_adj_user = np.reshape(1, tr_adj_event.shape[0], tr_adj_event.shape[1])
    tr_adj_user = torch.unsqueeze((torch.from_numpy(tr_adj_user)), dim=0)
    tr_adj_event = torch.unsqueeze((torch.from_numpy(tr_adj_event)), dim=0)
    # tr_adj_week_user_event = torch.unsqueeze((torch.from_numpy(tr_adj_week_user_event)), dim=0)
    # tr_adj_week_event_user = torch.unsqueeze((torch.from_numpy(tr_adj_week_event_user)), dim=0)
    tr_adj_dis_user_event = torch.unsqueeze((torch.from_numpy(tr_adj_dis_user_event)), dim=0)
    tr_adj_dis_event_user = torch.unsqueeze((torch.from_numpy(tr_adj_dis_event_user)), dim=0)
    tr_adj_time_week_user_event = torch.unsqueeze((torch.from_numpy(tr_adj_time_week_user_event)), dim=0)
    tr_adj_time_week_event_user = torch.unsqueeze((torch.from_numpy(tr_adj_time_week_event_user)), dim=0)
    # 训练集用户的邻居元素
    train_user_matrix = torch.cat((tr_adj_user, tr_adj_dis_user_event, tr_adj_time_week_user_event))
    # 训练集事件的邻居元素
    train_event_matrix = torch.cat(
        (tr_adj_event, tr_adj_dis_event_user, tr_adj_time_week_event_user))

    # 验证集
    # val_adj_user, val_adj_event, val_adj_week_user_event, val_adj_week_event_user, val_adj_dis_user_event, val_adj_dis_event_user = getData(
    #     value_ratings)
    # val_adj_user = torch.unsqueeze((torch.from_numpy(val_adj_user)), dim=0)
    # val_adj_event = torch.unsqueeze((torch.from_numpy(val_adj_event)), dim=0)
    # val_adj_week_user_event = torch.unsqueeze((torch.from_numpy(val_adj_week_user_event)), dim=0)
    # val_adj_week_event_user = torch.unsqueeze((torch.from_numpy(val_adj_week_event_user)), dim=0)
    # val_adj_dis_user_event = torch.unsqueeze((torch.from_numpy(val_adj_dis_user_event)), dim=0)
    # val_adj_dis_event_user = torch.unsqueeze((torch.from_numpy(val_adj_dis_event_user)), dim=0)
    # # 验证集用户的邻居元素
    # val_user_matrix = torch.cat(
    #     (val_adj_user, val_adj_week_user_event, val_adj_dis_user_event))
    # # 验证集事件的邻居元素
    # val_event_matrix = torch.cat(
    #     (val_adj_event, val_adj_week_event_user, val_adj_dis_event_user))

    # 测试集
    # te_adj_user, te_adj_event, te_adj_week_user_event, te_adj_week_event_user, te_adj_dis_user_event, te_adj_dis_event_user = getData(test_ratings)
    # te_adj_user = torch.unsqueeze((torch.from_numpy(te_adj_user)), dim=0)
    # te_adj_event = torch.unsqueeze((torch.from_numpy(te_adj_event)), dim=0)
    # te_adj_week_user_event = torch.unsqueeze((torch.from_numpy(te_adj_week_user_event)), dim=0)
    # te_adj_week_event_user = torch.unsqueeze((torch.from_numpy(te_adj_week_event_user)), dim=0)
    # te_adj_dis_user_event = torch.unsqueeze((torch.from_numpy(te_adj_dis_user_event)), dim=0)
    # te_adj_dis_event_user = torch.unsqueeze((torch.from_numpy(te_adj_dis_event_user)), dim=0)
    # # 验证集用户的邻居元素
    # te_user_matrix = torch.cat(
    #     (te_adj_user, te_adj_week_user_event, te_adj_dis_user_event))
    # te_event_matrix = torch.cat(
    #     (te_adj_event, te_adj_week_event_user, te_adj_dis_event_user))
    #

    # 获得训练集的lable
    # GetLable = Lable()
    # lable = GetLable.getNum(trian_ratings)

    return trian_ratings, train_user_matrix, train_event_matrix

if __name__ == '__main__':

    train_ratings, train_user_matrix, train_event_matrix = data_partition("./data")
    labler = Lable()
    lable = labler.getLable_(train_ratings)