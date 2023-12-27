# training settings
batch_size = 64
lr = 0.001  # learning rate for the generator
epochs = 20
gru_epochs = 100

at_header_num = 6

# model saving
load_mdel = False  # whether loading existing model for initialization

# other hypoer-parameters
n_emb = 105
n_hidden = 32
n_out = 95
dense_hidden1 = 32
dense_hidden2 = 16
dense_out = 1
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 5

in_channels = 2
out_channels = 1

w_out = 50
# path settings
data_filename = "infoToNum.csv"
save_path = "../data/words_embed.txt"
glove_path = "../data/glove.6B.50d.txt"
glove_txt_path = "../data/glove.txt"

#  cut_len
cut_len = 20

heads = 8

neighbor_num = 5

# candidate_num
candidate_num = 100
