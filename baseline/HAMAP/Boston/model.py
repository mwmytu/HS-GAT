import torch
import torch.nn as nn
import math
import config
import torch.nn.functional as F
# from test import getData
from torch_geometric.nn import GCNConv


# 语义级别的注意力
class OneGraphAttention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(OneGraphAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # nn.Linear(hidden_size, 5),
        )

    def forward(self, z):
        # 求出每个元路径下每个元路径对应的注意力权重
        w = self.project(z)  # (M, 1)
        # w = torch.sum(w, dim=-1)
        # 求出每个元路径的注意力系数
        # 通过softmax可以求出某个权值的重要程度
        beta = torch.softmax(w, dim=0)  # (M, 1)
        # beta = torch.unsqueeze(beta, dim=-1)
        # beta = beta.expand((z.shape[0],) +beta.shape)  # (N, M, 1)
        # embed = torch.mm(beta.T, z)
        # 返回最终的经过语义处理的每个节点的embed
        return beta  # (N, D * K)


class HAMAP(nn.Module):
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
        super(HAMAP, self).__init__()

        self.user_num = user_num
        self.poi_num = item_num
        self.embed_num = embed_dim
        self.gcn_embed_dim = gcn_embed_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = out_dim
        self.edge_user = edge_user
        self.at_header_num = at_header_num

        # 用户编码映射层
        self.user_embed = nn.Embedding(self.user_num+1, self.embed_num)
        # POI编码映射层
        self.poi_embed = nn.Embedding(self.poi_num+1, self.embed_num)
        # 用户GCN层
        self.gcn1_user = GCNConv(self.embed_num, self.gcn_embed_dim)
        self.gcn2_user = GCNConv(self.gcn_embed_dim, self.gcn_embed_dim)
        # POI GCN层
        self.gcn1_poi = GCNConv(self.embed_num, self.gcn_embed_dim)
        self.gcn2_poi = GCNConv(self.gcn_embed_dim, self.gcn_embed_dim)
        # 用户 Linear层
        self.dense1_user = nn.Linear(config.n_emb, self.hidden_dim_1)
        self.dense2_user = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.dense3_user = nn.Linear(self.hidden_dim_2, self.poi_num)
        # Check-ins 映射层
        self.check_ins_embed = nn.Embedding((int(edge_user[1].max() + 1)), self.embed_num)
        self.check_ins_embed2 = nn.Embedding((int(edge_user[1].max() + 1)), self.embed_num)

        self.at_multi_list, self.at_multi_list2 = [], []
        for i in range(self.at_header_num):
            self.at_multi_list.append(OneGraphAttention(self.embed_num))
            self.at_multi_list2.append(OneGraphAttention(self.embed_num))
        # poi嵌入升维
        self.dense1_poi = nn.Linear(self.embed_num, self.embed_num)
        w = torch.Tensor(1, 49)
        self.w1 = torch.nn.init.normal_(w, mean=0, std=0.001)
        w2 = torch.Tensor(config.n_emb, self.poi_num)
        self.w2 = torch.nn.init.normal_(w2, mean=0, std=0.001)

        self.dense1_ = nn.Linear(config.n_emb, 120)
        self.dense2_ = nn.Linear(49, 1)

    def forward(self, edge_user, edge_poi, u_v_edge, v_u_edge):
        # 获得用户的初始嵌入
        user_embed = self.user_embed(torch.tensor([i for i in range(self.user_num)]))
        # 获得poi的初始嵌入
        poi_embed = self.poi_embed(torch.tensor([i for i in range(self.poi_num)]))
        # gcn更新嵌入
        user_embed = self.gcn1_user(user_embed, torch.tensor(u_v_edge).long())
        user_embed = F.relu(user_embed)
        user_embed = F.dropout(user_embed, p=0.2)
        user_embed = self.gcn2_user(user_embed, torch.tensor(u_v_edge).long())

        poi_embed = self.gcn1_poi(poi_embed, torch.tensor(v_u_edge).long())
        poi_embed = F.relu(poi_embed)
        poi_embed = F.dropout(poi_embed, p=0.2)
        poi_embed = self.gcn2_poi(poi_embed, torch.tensor(v_u_edge).long())
        # check-ins 嵌入
        check_ins_embed = self.check_ins_embed(torch.reshape(torch.tensor(edge_user[1]).long(), (1, -1)))
        check_ins_embed = torch.reshape(check_ins_embed,
                                      (1, edge_user.shape[1], edge_user.shape[2], self.embed_num))
        check_ins_embed = torch.squeeze(check_ins_embed)
        check_ins_embed2___ = self.check_ins_embed2(torch.reshape(torch.tensor(edge_user[1]).long(), (1, -1)))
        check_ins_embed2___ = torch.reshape(check_ins_embed2___,
                                        (1, edge_user.shape[1], edge_user.shape[2], self.embed_num))
        check_ins_embed2___ = torch.squeeze(check_ins_embed2___)
        times = 0
        user_check_ins_embed_, user_check_ins_embed_2 = None, None
        for j in range(self.at_header_num):
            user_check_ins_embed = torch.zeros((check_ins_embed.shape[0], 5, 8))
            for i in range(check_ins_embed.shape[0]):
                user_check_ins_embed[i] = self.at_multi_list[j](check_ins_embed[i])
            if times == 0:
                user_check_ins_embed_ = user_check_ins_embed
            else:
                user_check_ins_embed_ = torch.cat((user_check_ins_embed_, user_check_ins_embed), dim=-1)
            times += 1
        # user_check_ins_embed_ = torch.reshape(user_check_ins_embed_, (user_check_ins_embed_.shape[0], -1))
        user_check_ins_embed_ = user_check_ins_embed_.permute((0, 2, 1))
        use_trans_embed = torch.matmul(user_check_ins_embed_, check_ins_embed)
        user_embed = torch.reshape(user_embed, (user_embed.shape[0], 1, user_embed.shape[-1]))
        user_embed_final = torch.cat((user_embed, use_trans_embed), dim=1)
        user_embed_final_ = torch.zeros((check_ins_embed.shape[0], config.n_emb, 1))
        user_embed_final = user_embed_final.permute((0, 2, 1))
        for i in range(user_embed_final.shape[0]):
            user_embed_final_[i] = self.dense2_(user_embed_final[i])
        user_embed_final = torch.squeeze(user_embed_final_)
        # user_embed_final = torch.squeeze(torch.matmul(self.w1, user_embed_final))

        # mlp 聚合用户embed
        user_embed = F.dropout(F.relu(self.dense1_user(user_embed_final)), 0.2)
        user_embed = F.dropout(F.relu(self.dense2_user(user_embed)), 0.2)
        user_score1 = self.dense3_user(user_embed)

        user_check_ins_embed2 = torch.zeros((check_ins_embed.shape[0], 5, 120))
        for i in range(check_ins_embed.shape[0]):
            user_check_ins_embed2[i] = self.dense1_(check_ins_embed2___[i])
        check_ins_embed = torch.matmul(self.w2.T, check_ins_embed.permute((0, 2, 1)))
        user_score2 = torch.matmul(check_ins_embed, user_check_ins_embed_.permute((0, 2, 1)))
        # user_score2 = torch.matmul(check_ins_embed.permute((0, 2, 1)), user_check_ins_embed_)
        # user_score2 = torch.sum(user_score2.permute((0, 2, 1)), dim=1)
        # score_final = torch.sigmoid(user_score1 + user_score2)
        user_score2 = torch.sum(user_check_ins_embed2, dim=1)
        score_final = user_score1 + user_score2
        return score_final
