import datetime
from torch import nn
import math

import config
from utils import *
from dataloader import getData, train_deal, test_deal
from model import Transformer

# 指定gpu设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ratings, train_user_matrix, train_event_matrix = data_prepare.data_partition(fname='./data')
dataset, train_data, test_data, ug_u_u2, ug_v_v2 = getData()
u_v_matrix, v_u_matrix, target, words_list, u_u_dict, v_v_dict, u_u_dict_all, u_v_star = train_deal(dataset, train_data)

# Sample Usage
src_pad_idx = 0
trg_pad_idx = 0
trg_sos_idx = 1
enc_voc_size = 200
dec_voc_size = 200
d_model = 50
n_head = 5
max_len = 1000
ffn_hidden = 128
n_layers = 5
drop_prob = 0.1
device = torch.device('cpu')

# Create an instance of the Transformer model
model = Transformer(src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model,
                                n_head, max_len, ffn_hidden, n_layers, drop_prob, device, v_u_matrix.shape[1])

# Move the model to the desired device
model = model.to(device)
# model.load_state_dict(torch.load("./models/2023-11-22_23_45_43.pth"))
model.load_state_dict(torch.load("./models/2023-11-23_00_08_59.pth"))
mse = nn.MSELoss(reduction='mean')
user_id_matrix = np.zeros((1, u_v_matrix.shape[1], u_v_matrix.shape[2]))
for i in range(user_id_matrix.shape[1]):
    user_id_matrix[0][i] = np.array([i for val in range(user_id_matrix.shape[2])])
print(user_id_matrix)
matrix = torch.tensor(np.concatenate((u_v_matrix, user_id_matrix), axis=0), dtype=torch.long).permute(1, 2, 0)
matrix = torch.reshape(matrix, (matrix.shape[0], -1)).to(device)
# Create sample input tensors
src_input = torch.randint(0, enc_voc_size, (32, 50)).to(device)  # Batch size: 32, Sequence length: 50
trg_input = torch.tensor(u_v_star, dtype=torch.long).to(device)  # Batch size: 32, Sequence length: 60
target = torch.tensor(target, dtype=torch.float32)
u_i_dict = test_deal(dataset, test_data)
with torch.no_grad():
    for q in range(100):
        print("-------------")
        aver_recall_list, aver_pre_list, aver_f1_list = [], [], []
        for i in range(50):
            output = model(matrix, trg_input)
            mse_loss = mse(output, target)
            rmse_loss = math.sqrt(mse_loss)
            output_ = output.detach().numpy()
            # 根据留一法获得测试结果
            test_topn_list2d = []
            for k in u_i_dict.keys():
                topn_list1d = []
                for j in u_i_dict[k]:
                    topn_list1d.append(output_[k][j])
                test_topn_list2d.append(topn_list1d)
            test_topn_arr2d = np.array(test_topn_list2d)
            N = 1
            hits_list = []
            # ndcg_aver_list = []
            # mrr_list = []
            recall_list = []
            precision_list = []
            f1_list = []
            for j in range(20):
                hits_, hits_dict = hits(u_i_dict, test_topn_arr2d, N)
                hits_list.append(hits_)

                # ndcg_ = get_ndcg(hits_dict, u_i_dict)
                # ndcg_aver_list.append(ndcg_)
                # mrr_ = mrr(hits_dict, u_i_dict)
                # mrr_list.append(mrr_)
                recall_list.append(recall_topn(hits_))
                precision_list.append(precision_topn(hits_, N))
                f1_list.append(f1_score_topn(recall_topn(hits_), precision_topn(hits_, N)))

                N += 1

            aver_recall_list.append(recall_list)
            aver_pre_list.append(precision_list)
            aver_f1_list.append(f1_list)

        aver_recall_list_, aver_pre_list_, aver_f1_list_ = [], [], []
        for i in range(len(aver_recall_list[0])):
            recall_sum, pre_sum, f1_sum = 0, 0, 0
            for j in range(len(aver_recall_list)):
                recall_sum += aver_recall_list[j][i]
                pre_sum += aver_pre_list[j][i]
                f1_sum += aver_f1_list[j][i]
            aver_recall_list_.append(round(recall_sum / len(aver_recall_list), 5))
            aver_pre_list_.append(round(pre_sum / len(aver_pre_list), 5))
            aver_f1_list_.append(round(f1_sum / len(aver_f1_list), 5))

        print("aver recall:{}".format(aver_recall_list_))
        print("aver precision:{}".format(aver_pre_list_))
        print("aver f1-score:{}".format(aver_f1_list_))

        print("mse:{}".format(mse_loss))
        print("rmse:{}".format(rmse_loss))
