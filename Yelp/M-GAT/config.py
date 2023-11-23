# training settings
batch_size = 64
lr = 1e-3  # learning rate for the generator
epochs = 300
gru_epochs = 100

# model saving
load_model = False  # whether loading existing model for initialization

# other hyper-parameters
n_emb = 20
n_hidden = 36
n_out = 20
dense_hidden1 = 16
dense_hidden2 = 8
dense_out = 1
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 5

in_channels = 3
out_channels = 1

w_out = 20

# path settings
data_filename = "infoToNum.csv"
save_path = "../../data/words_embed.txt"
glove_path = "../../data/glove.6B.50d.txt"
glove_txt_path = "../../data/glove.txt"

#  cut_len
cut_len = 20

heads = 8

neighbor_num = 5

# candidate_num
candidate_num = 100
