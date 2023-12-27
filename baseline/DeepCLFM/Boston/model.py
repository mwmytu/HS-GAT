import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from d2l import torch as d2l

from config import out_dim


# 语义级别的注意力
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()

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
        beta = torch.squeeze(torch.softmax(w, dim=0))  # (M, 1)
        # beta = beta.expand((z.shape[0],) +beta.shape)  # (N, M, 1)
        # 返回最终的经过语义处理的每个节点的embed
        return beta  # (N, D * K)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = tokens_a
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a))
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


# 因子分解
class FM(nn.Module):
    def __init__(self, n):
        super(FM, self).__init__()
        # 特征的个数
        self.n = n
        # 隐向量的维度
        self.k = out_dim
        self.linear = nn.Linear(self.n, out_dim, bias=True)
        self.V = nn.Parameter(torch.randn(self.n, self.k), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        linear = self.linear(x)
        # 乘积的平方减去平方的积
        inter = 0.5 * torch.sum(torch.pow(torch.mm(x, self.V), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.V, 2)), dim=(1, ), keepdim=True)
        output = linear + inter
        logit = self.sigmoid(output)
        return logit


class DeepCLFM(nn.Module):
    def __init__(self, user_num, item_num):
        super(DeepCLFM, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        # 注意力机制
        self.user_self_attention = Attention(out_dim)
        self.item_self_attention = Attention(out_dim)

        self.u_second_fm = FM(out_dim*2)
        self.i_second_fm = FM(out_dim*2)
        self.lfm_u_fm = FM(out_dim)
        self.lfm_i_fm = FM(out_dim)
        self.user_gru = nn.GRU(out_dim, out_dim, num_layers=2, bidirectional=True)
        self.item_gru = nn.GRU(out_dim, out_dim, num_layers=2, bidirectional=True)

        self.linear1 = nn.Linear(out_dim*4, 64)
        self.linear2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, user_words_bert_tensor3d, item_words_bert_tensor3d, LFM_U: Tensor, LFM_I: Tensor, u_u_dict):
        user_words_bert_tensor3d = torch.permute(user_words_bert_tensor3d, (1, 0, 2))
        user_out, user_hid = self.user_gru(user_words_bert_tensor3d)
        user_text_embed = user_hid[-1]
        # 获得每个用户的贡献程度
        beta = self.user_self_attention(user_text_embed)
        for i in range(user_text_embed.shape[0]):
            user_text_embed[i] = user_text_embed[i]*beta[i]
        TEXT_U = user_text_embed

        # 获得项目文本embed
        item_words_bert_tensor3d = torch.permute(item_words_bert_tensor3d, (1, 0, 2))
        item_out, item_hid = self.item_gru(item_words_bert_tensor3d)
        item_text_embed = item_hid[-1]
        # 获得每个项目的贡献程度
        beta = self.item_self_attention(item_text_embed)
        for i in range(item_text_embed.shape[0]):
            item_text_embed[i] = item_text_embed[i]*beta[i]
        TEXT_I = item_text_embed

        # 对embed进行求和
        U_First = TEXT_U + LFM_U
        I_First = TEXT_I + LFM_I

        # 进行因子分解计算
        TEXT_LFM_U = torch.concat((user_text_embed, LFM_U), dim=-1)
        TEXT_LFM_I = torch.concat((item_text_embed, LFM_I), dim=-1)
        U_Second = self.u_second_fm(TEXT_LFM_U)
        I_Second = self.i_second_fm(TEXT_LFM_I)

        # 进行拼接
        U = torch.concat((U_First, U_Second), dim=-1)
        I = torch.concat((I_First, I_Second), dim=-1)
        # 将用户embed和事件embed进行拼接
        u_v_embed_cat = []
        # for k, v in enumerate(u_u_dict):
        #     for j in range(len(u_u_dict[v])):
        #         u_v_embed_cat.append(torch.cat((U[v], I[u_u_dict[v][j]])).tolist())
        # u_v_embed_cat = torch.tensor(u_v_embed_cat)

        for i in range(U.shape[0]):
            for j in range(I.shape[0]):
                u_v_embed_cat.append(torch.cat((U[i], I[j])).tolist())
        u_v_embed_cat = torch.tensor(u_v_embed_cat)
        # 线性层
        X = F.dropout(F.leaky_relu(self.linear1(u_v_embed_cat)), p=0.3)
        X = F.dropout(F.leaky_relu(self.linear2(X)), p=0.3)
        score = F.leaky_relu(self.out(X))
        score = torch.reshape(score, (U.shape[0], I.shape[0]))

        return score


