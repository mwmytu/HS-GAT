import torch
import torch.nn as nn
import math
import config
import torch.nn.functional as F
# from test import getData
from torch_geometric.nn import GCNConv


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


class ATST_LSTM(nn.Module):
    def __init__(
            self,
            user_num: int,
            item_num: int,
            embed_dim: int,
            gcn_embed_dim: int,
            hidden_dim_1: int,
            hidden_dim_2: int,
            out_dim: int,
            edge_user,
            at_header_num: int
    ):
        super(ATST_LSTM, self).__init__()

        self.user_num = user_num
        self.poi_num = item_num
        self.embed_num = embed_dim
        self.gcn_embed_dim = gcn_embed_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = out_dim
        self.edge_user = edge_user
        self.at_header_num = at_header_num

        # 用户 Linear层
        self.dense1_user = nn.Linear(self.gcn_embed_dim, self.hidden_dim_1)
        self.dense2_user = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.dense3_user = nn.Linear(self.hidden_dim_2, self.poi_num)

        self.id_embed = nn.Embedding((int(edge_user[0].max() + 1)), self.embed_num)
        self.p_embed = nn.Embedding((int(edge_user[1].max() + 1)), self.embed_num)
        self.t_embed = nn.Embedding((int(edge_user[2].max() + 1)), self.embed_num)

        self.oneGraphAttention_user = OneGraphAttention(embed_dim)

        self.lstm = nn.LSTM(input_size=self.embed_num*3, hidden_size=config.n_emb, num_layers=2, batch_first=False)

        self.at_multi_list = []
        for i in range(self.at_header_num):
            self.at_multi_list.append(OneGraphAttention(self.embed_num))
        # poi嵌入升维
        self.dense1_poi = nn.Linear(self.embed_num, self.embed_num*self.at_header_num)

    def forward(self, edge_user, edge_poi, u_v_edge, v_u_edge):
        id_embed = self.id_embed(torch.reshape(torch.tensor(edge_user[0]).long(), (1, -1)))
        p_embed = self.p_embed(torch.reshape(torch.tensor(edge_user[1]).long(), (1, -1)))
        t_embed = self.t_embed(torch.reshape(torch.tensor(edge_user[2]).long(), (1, -1)))
        id_embed = torch.reshape(id_embed,
                                      (1, edge_user.shape[1], edge_user.shape[2], self.embed_num))
        id_embed = torch.squeeze(id_embed)
        p_embed = torch.reshape(p_embed,
                                 (1, edge_user.shape[1], edge_user.shape[2], self.embed_num))
        p_embed = torch.squeeze(p_embed)
        t_embed = torch.reshape(t_embed,
                                 (1, edge_user.shape[1], edge_user.shape[2], self.embed_num))
        t_embed = torch.squeeze(t_embed)
        final_embed = torch.cat((id_embed, p_embed, t_embed), dim=-1)
        # final_embed = final_embed.permute(1, 0, 2)
        out = self.lstm(final_embed)
        user_embed = torch.zeros((edge_user.shape[1], self.embed_num))
        for i in range(out[0].shape[0]):
            # self.user_embed2[i] = self.oneGraphAttention_user(user_event_embed[i])
            user_embed[i] = self.oneGraphAttention_user(out[0][i])
        user_embed = F.dropout(F.relu(self.dense1_user(user_embed)), 0.2)
        user_embed = F.dropout(F.relu(self.dense2_user(user_embed)), 0.2)
        user_score1 = self.dense3_user(user_embed)
        # print(user_score1)

        return user_score1
