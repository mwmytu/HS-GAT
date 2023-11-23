import torch
import torch.nn as nn
import math
import config
import torch.nn.functional as F
# from test import getData
from torch_geometric.nn import GCNConv, GATConv

#  = torch.("cuda:0" if torch.cuda.is_available() else "cpu")


# 语义级别的注意力
class OneGraphAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(OneGraphAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, z):
        # 求出每个元路径下每个元路径对应的注意力权重
        w = self.project(z)  # (M, 1)
        # 求出每个元路径的注意力系数
        # 通过softmax可以求出某个权值的重要程度
        beta = torch.softmax(w, dim=0)  # (M, 1)
        # beta = beta.expand((z.shape[0],) +beta.shape)  # (N, M, 1)
        embed = torch.mm(beta.T, z)
        # 返回最终的经过语义处理的每个节点的embed
        return embed  # (N, D * K)


class NN(nn.Module):
    def __init__(self, edge_user, edge_event, in_channels, out_channels, user_len, business_len, num_embed, w_out):
        super(NN, self).__init__()

        # 用户邻接矩阵
        self.edge_user = edge_user
        # 事件邻接矩阵
        self.edge_event = edge_event

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_user = edge_user.shape[1]
        self.num_business = edge_event.shape[1]
        self.user_len = user_len
        self.business_len = business_len
        self.num_emed = num_embed
        self.w_out = w_out

        # print(num_user, num_event, num_week, num_dis)

        # 用于存储用户-事件 事件-用户的embed
        self.user_embed2 = torch.zeros((user_len, self.num_emed))
        self.event_embed2 = torch.zeros((business_len, self.num_emed))
        # 初始化用户的embed
        self.user_mebed = nn.Embedding(self.num_user, self.num_business)
        # 初始化事件的embed
        self.event_embed = nn.Embedding(self.num_business, self.num_user)
        # 初始化用户事件交互的embed
        self.u_v_vid_embed = nn.Embedding((int(edge_user[0].max()+1)), num_embed)
        self.u_v_lati_longi_embed = nn.Embedding((int(edge_user[1].max()+1)), num_embed)
        self.u_v_item_stars_embed = nn.Embedding((int(edge_user[2].max()+1)), num_embed)
        # self.u_v_review_count_embed = nn.Embedding((int(edge_user[3].max()+1)), num_embed)
        # 初始化事件用户交互的embed
        self.v_u_uid_embed = nn.Embedding((int(edge_event[0].max()+1)), num_embed)
        self.v_u_date_embed = nn.Embedding((int(edge_event[1].max()+1)), num_embed)
        # self.v_u_yelping_since_embed = nn.Embedding((int(edge_event[1].max()+1)), num_embed)
        # self.v_u_user_review_count = nn.Embedding((int(edge_event[2].max()+1)), num_embed)

        # nn.init.normal_(self.user_mebed.weight, std=0.1)
        # nn.init.normal_(self.event_embed.weight, std=0.1)

        # self.GTLayer = GTLayer(self.in_channels, self.out_channels)
        self.conv_user = GTConv(self.in_channels, self.out_channels, num_embed)
        self.conv_event = GTConv(self.in_channels, self.out_channels, num_embed)

        # GCN
        self.gcn1_user = GCNConv(config.n_emb, config.n_hidden)
        self.gcn2_user = GCNConv(config.n_hidden, config.n_out)
        self.gcn1_business = GCNConv(config.n_emb, config.n_hidden)
        self.gcn2_business = GCNConv(config.n_hidden, config.n_out)
        # GAT
        self.gat1_user = GATConv(config.n_emb, config.n_hidden, heads=config.heads)
        self.gat2_user = GATConv(config.n_hidden * config.heads, config.n_out)
        self.gat1_business = GATConv(config.n_emb, config.n_hidden, heads=config.heads)
        self.gat2_business = GATConv(config.n_hidden * config.heads, config.n_out)

        self.gru = nn.GRU(input_size=50, hidden_size=20, num_layers=2, batch_first=False)

        # self.at_gru_user = nn.GRU(input_size=20, hidden_size=20, num_layers=2, batch_first=False)
        # self.at_gru_business = nn.GRU(input_size=20, hidden_size=20, num_layers=2, batch_first=False)

        # 单图级别的注意力
        self.oneGraphAttention_user = OneGraphAttention(num_embed)
        self.oneGraphAttention_event = OneGraphAttention(num_embed)

        self.dense1 = nn.Linear(config.n_out * 2, config.dense_hidden1)
        self.dense2 = nn.Linear(config.dense_hidden1, config.dense_hidden2)
        self.out = nn.Linear(config.dense_hidden2, config.dense_out)

    def compute_embedding(self, words_embed, edge_user, edge_event, v_v_dict):
        # 映射各个矩阵的embed
        # 用户事件交互
        edge_user = torch.tensor(edge_user)
        u_v_vid_embed = torch.reshape(self.u_v_vid_embed(torch.reshape(edge_user[0].long(), (1, -1))), (1, edge_user.shape[1], edge_user.shape[2], self.num_emed))
        u_v_lati_longi_embed = torch.reshape(self.u_v_lati_longi_embed(torch.reshape(edge_user[1].long(), (1, -1))), (1, edge_user.shape[1], edge_user.shape[2], self.num_emed))
        u_v_item_starts_embed = torch.reshape(self.u_v_item_stars_embed(torch.reshape(edge_user[2].long(), (1, -1))), (1, edge_user.shape[1], edge_user.shape[2], self.num_emed))
        # u_v_review_count_embed = torch.reshape(self.u_v_review_count_embed(torch.reshape(edge_user[3].long(), (1, -1))), (1, edge_user.shape[1], edge_user.shape[2], self.num_emed))

        # 事件用户交互
        edge_event = torch.tensor(edge_event)
        v_u_uid_embed = torch.reshape(self.v_u_uid_embed(torch.reshape(edge_event[0].long(), (1, -1))), (1, edge_event.shape[1], edge_event.shape[2], self.num_emed))
        v_u_date_embed = torch.reshape(self.v_u_date_embed(torch.reshape(edge_event[1].long(), (1, -1))), (1, edge_event.shape[1], edge_event.shape[2], self.num_emed))
        # v_u_yelping_since_embed = torch.reshape(self.v_u_yelping_since_embed(torch.reshape(torch.tensor(self.edge_event[1]).long(), (1, -1))), (1, self.edge_event.shape[1], self.edge_event.shape[2], self.num_emed))
        # v_u_user_review_count = torch.reshape(self.v_u_user_review_count(torch.reshape(edge_event[2].long(), (1, -1))), (1, edge_event.shape[1], edge_event.shape[2], self.num_emed))
        v_u_words_embed = torch.zeros((edge_event.shape[1], edge_event.shape[2], words_embed.shape[1]))

        l1 = 0
        # for k, (v, i) in enumerate(v_v_dict):
        for k, v in enumerate(v_v_dict):
            l2 = 0
            # i = 0
            for j in range(len(v_v_dict[v])):
                v_u_words_embed[v][j] = words_embed[j+l1]
                l2 += 1
                # i += 1
            l1 += l2

        v_u_words_embed = v_u_words_embed.unsqueeze(dim=0)

        # 将用户-事件和事件-用户的矩阵根据维度进行合并
        # user_event_embed = torch.cat((u_v_vid_embed, u_v_lati_longi_embed, u_v_item_starts_embed, u_v_review_count_embed), dim=0)
        user_event_embed = torch.cat((u_v_vid_embed, u_v_lati_longi_embed, u_v_item_starts_embed), dim=0)
        # user_event_embed = torch.cat((u_v_lati_longi_embed, u_v_item_starts_embed, u_v_review_count_embed), dim=0)

        # event_user_embed = torch.cat((v_u_uid_embed, v_u_date_embed, v_u_user_review_count, v_u_words_embed), dim=0)
        event_user_embed = torch.cat((v_u_uid_embed, v_u_date_embed, v_u_words_embed), dim=0)
        # user_event_embed = torch.cat((u_v_vid_embed, u_v_lati_longi_embed, u_v_item_starts_embed, u_v_review_count_embed), dim=0)
        # event_user_embed = torch.cat((v_u_date_embed, v_u_user_review_count, v_u_words_embed), dim=0)

        return user_event_embed, event_user_embed

    def forward(self, words_embed, u_v_matrix, v_u_matrix, ug_u_u2, ug_v_v2, u_u_dict, v_v_dict):

        # 获得语义级别的embed
        words_embed = words_embed.permute(1, 0, 2)
        out, (h, c) = self.gru(words_embed)
        # g2 = self.tanh2(self.gru2(g1))
        words_embed = h
        # words_embed = words_embed
        # event_user_embed = event_user_embed

        user_event_embed, event_user_embed = self.compute_embedding(words_embed, u_v_matrix, v_u_matrix, v_v_dict)
        user_event_embed = self.conv_user(user_event_embed)
        event_user_embed = self.conv_event(event_user_embed)

        user_embed = torch.zeros((self.user_len, self.num_emed))
        event_embed = torch.zeros((self.business_len, self.num_emed))

        # 初始化用户和事件embed完成
        for i in range(user_event_embed.shape[0]):
            # self.user_embed2[i] = self.oneGraphAttention_user(user_event_embed[i])
            user_embed[i] = self.oneGraphAttention_user(user_event_embed[i])
        # user_embed = user_embed.requires_grad_(True)

        for i in range(event_user_embed.shape[0]):
            # self.event_embed2[i] = self.oneGraphAttention_event(event_user_embed[i])
            event_embed[i] = self.oneGraphAttention_event(event_user_embed[i])
        # event_embed = event_embed.requires_grad_(True)

        # 根据用户和事件初始化embed和各邻接矩阵  进行GCN操作
        # user_embed = user_embed
        # event_embed = event_embed
        x_u = self.gcn1_user(user_embed, torch.tensor(ug_u_u2).long())
        x_u = F.relu(x_u)
        x_u = F.dropout(x_u, p=0.3)
        x_u = self.gcn2_user(x_u, torch.tensor(ug_u_u2).long())

        x_v = self.gcn1_business(event_embed, torch.tensor(ug_v_v2).long())
        x_v = F.relu(x_v)
        x_v = F.dropout(x_v, p=0.3)
        x_v = self.gcn2_business(x_v, torch.tensor(ug_v_v2).long())

        # GAT
        # x_u = self.gat1_user(user_embed, torch.tensor(ug_u_u2).long())
        # x_u = F.relu(x_u)
        # # x_u = F.leaky_relu(x_u)
        # x_u = F.dropout(x_u, p=0.3)
        # x_u = self.gat2_user(x_u, torch.tensor(ug_u_u2).long())
        #
        # x_v = self.gat1_business(event_embed, torch.tensor(ug_v_v2).long())
        # x_v = F.relu(x_v)
        # # x_v = F.leaky_relu(x_v)
        # x_v = F.dropout(x_v, p=0.3)
        # x_v = self.gat2_business(x_v, torch.tensor(ug_v_v2).long())

        # 将用户embed和事件embed进行拼接
        # u_v_embed_cat = []
        # # for i in range(x_u.shape[0]):
        # #     for j in range(x_v.shape[0]):
        # #         u_v_embed_cat.append(torch.cat((x_u[i], x_v[j])).tolist())
        # # u_v_embed_cat = torch.tensor(u_v_embed_cat)
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat.append(torch.cat((x_u[v], x_v[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat = torch.tensor(u_v_embed_cat)
        # # u_v_embed_cat = u_v_embed_cat.requires_grad_(True)
        # # for i in range(x_u.shape[0]):
        # #     for j in range(x_v.shape[0]):
        # #         u_v_embed_cat.append(torch.cat((x_u[i], x_v[j])).tolist())
        # # u_v_embed_cat = torch.tensor(u_v_embed_cat)
        # # u_v_embed_cat = u_v_embed_cat.requires_grad_(True)
        #
        # x = F.dropout(F.relu(self.dense1(u_v_embed_cat)), p=0.3)
        # # x = F.dropout(F.leaky_relu(self.dense1(u_v_embed_cat)), p=0.3)
        # x = F.dropout(F.relu(self.dense2(x)), p=0.3)
        # # x = F.dropout(F.leaky_relu(self.dense2(x)), p=0.3)
        # # score = F.softmax(self.out(x))
        # # score = F.sigmoid(self.out(x))
        # score = F.relu(self.out(x))
        score = torch.matmul(x_u, x_v.T)

        return score


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first

        self.conv_user = GTConv(in_channels, out_channels)
        self.conv_event = GTConv(in_channels, out_channels)

    def forward(self, A_user, A_event):
        A_user = self.conv_user(A_user)
        A_event = self.conv_event(A_event)

        return A_user, A_event

# GTN网络层
class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_dim):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1, num_dim))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

        # self.leaky_relu = nn.LeakyReLU()
        # self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        # A = A
        A = A * F.softmax(self.weight, dim=1)
        A = torch.sum(torch.squeeze(A, dim=0), dim=0)

        # 最终返回已经求和之后的Q
        # A = self.leaky_relu(self.cnn(A))
        return A
