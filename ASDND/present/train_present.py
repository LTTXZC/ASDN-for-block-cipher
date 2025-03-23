from train_nets_dialate import train_distinguisher
from present1 import Present
from tensorflow.keras.models import load_model

from make_train_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7
n_samples_mult_pairs = 10**6

# net = load_model('./dilated_models/Present_7_best_20230523_202611.h5')
# Script for training a Present distinguisher using the same hyper-parameter as in the paper

present = Present(n_rounds=8)
print(present.n_rounds)
train_distinguisher(
    present, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xd, 0, 0, 0, 0, 0], kernel_size=5,depth=3,n_epochs=50,lr_high=0.003, lr_low=0.00028,
    reg_param=0.00000062)

# #
in_diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xd, 0, 0, 0, 0, 0]

# if __name__ == "__main__":
#     print("### Combining scores of Present neural distinguishers under independence assumption ###")
#     print("8 rounds:")
#     for pairs in [1, 2, 4, 8, 16,32,64]:
#         print(f'{pairs} pairs:')
#         x, y = make_mult_pairs_data(n_samples_mult_pairs, present, in_diff, n_pairs=pairs)
#         evaluate_mult_pairs(net, present, x, y, n_pairs=pairs)