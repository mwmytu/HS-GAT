import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import numpy as np

from dataloader import get_data, train_deal, test_deal
from config import out_dim


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = tokens_a
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a))
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTEncoder(nn.Module):
    """BERT编码器"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=22000, key_size=out_dim, query_size=out_dim, value_size=out_dim,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        # 将transformer中的EncodeerBlosck照搬过来
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数  随机初始化
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度 ，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[0], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


if __name__ == '__main__':
    # 获得数据集
    dataset, train_data, test_data = get_data()
    target, u_u_dict, train_data = train_deal(dataset, train_data)
    # 获得用户-项目评论集
    user_words_df = pd.read_csv("./word_data/user_words_str_40000.csv", encoding="utf8")["words"]
    item_words_df = pd.read_csv("./word_data/item_words_str_40000.csv", encoding="utf8")["words"]

    user_words_list2d = []
    for i in range(user_words_df.shape[0]):
        user_words_list2d.append([int(j) for j in user_words_df.loc[i].split(",")])
    user_words_list1d = np.squeeze(np.reshape(np.array(user_words_list2d), (-1, 1))).tolist()
    item_words_list2d = []
    for i in range(item_words_df.shape[0]):
        item_words_list2d.append([int(j) for j in item_words_df.loc[i].split(",")])
    item_words_list1d = np.squeeze(np.reshape(np.array(item_words_list2d), (-1, 1))).tolist()
    # 用户
    user_tokens, user_segments = get_tokens_and_segments(user_words_list1d)
    # 项目
    item_tokens, item_segments = get_tokens_and_segments(item_words_list1d)
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 22000, out_dim, 8, 5
    norm_shape, ffn_num_input, num_layers, dropout = [out_dim], out_dim, 1, 0.5
    encode = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                         ffn_num_hiddens, num_heads, num_layers, dropout)

    user_encode = encode(torch.tensor(user_tokens, dtype=torch.int), torch.tensor(user_segments, dtype=torch.int), None)
    item_encode = encode(torch.tensor(item_tokens, dtype=torch.int), torch.tensor(item_segments, dtype=torch.int), None)
    user_encode = torch.squeeze(user_encode).detach().numpy().tolist()
    item_encode = torch.squeeze(item_encode).detach().numpy().tolist()
    user_text_embed_list2d = []
    for i in range(len(user_encode)):
        user_text_embed_list2d.append(",".join([str(j) for j in user_encode[i]]))
    item_text_embed_list2d = []
    for i in range(len(item_encode)):
        item_text_embed_list2d.append(",".join([str(j) for j in item_encode[i]]))
    user_bert_text_df = pd.DataFrame(user_text_embed_list2d, columns=["words"])
    item_bert_text_df = pd.DataFrame(item_text_embed_list2d, columns=["words"])
    # 保存bert预训练embed
    user_bert_text_df.to_csv("./word_data/user_bert_text_embed_15.csv", encoding="utf8")
    item_bert_text_df.to_csv("./word_data/item_bert_text_embed_15.csv", encoding="utf8")
