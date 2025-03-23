from train_nets_dialate import train_distinguisher
from skinny import Skinny
from tensorflow.keras.models import load_model
from eval import evaluate, evaluate_mult_pairs
from make_train_data import make_mult_pairs_data
# Script for training a Skinny distinguisher using the same hyper-parameter as in the paper
n_samples = 10**7
n_samples_mult_pairs = 10**6
# 单个区分器训练
skinny = Skinny(n_rounds=8)
print(skinny.n_rounds)
train_distinguisher(
    skinny, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], n_epochs=30, depth=5,calc_back=2, lr_high=0.0011, lr_low=0.000045,
    kernel_size=3, reg_param=0.000000043
)

# 多密文对
# skinny = Skinny(n_rounds=10)
# in_diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
# # load our distinguisher
# net = load_model('./skinny_model/Skinny_10_best_20230416_192044.h5')
# if __name__ == "__main__":
#     print("### Combining scores of Skinny neural distinguisher under independence assumption ###")
#     print('10 rounds:')
#     for pairs in [1, 2, 4, 8, 16,32,64]:
#         print(f'{pairs} pairs:')
#         x, y = make_mult_pairs_data(n_samples_mult_pairs, skinny, in_diff, calc_back=2, n_pairs=pairs)
#         evaluate_mult_pairs(net, skinny, x, y, n_pairs=pairs)