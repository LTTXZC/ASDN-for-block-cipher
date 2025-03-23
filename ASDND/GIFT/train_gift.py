from train_nets_dialate1 import train_distinguisher
from gift1 import Gift
from tensorflow.keras.models import load_model

from make_train_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs


gift = Gift(n_rounds=7)

print(gift.n_rounds)
train_distinguisher(
    gift, [0x0000,0x0000,0x0000,0x000a], n_epochs=50, depth=10,lr_high=0.001, lr_low=0.00028, kernel_size=3,
    reg_param=0.000000849)


# n_samples = 10**7
# n_samples_mult_pairs = 10**6
#
# net = load_model('./dilated_models/Gift_5_best_20230626_095046.h5')
# in_diff = [0x0000,0x0000,0x0000,0x000a]
#
# if __name__ == "__main__":
#     print("### Combining scores of Present neural distinguishers under independence assumption ###")
#     print("7 rounds:")
#     for pairs in [1, 2, 4, 8, 16,32,64]:
#         print(f'{pairs} pairs:')
#         x, y = make_mult_pairs_data(n_samples_mult_pairs, gift, in_diff, n_pairs=pairs)
#         evaluate_mult_pairs(net, gift, x, y, n_pairs=pairs)