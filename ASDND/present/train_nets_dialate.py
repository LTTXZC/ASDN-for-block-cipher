import numpy as np
from pickle import dump
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SeparableConv1D,Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

from tensorflow.keras import layers, Model, Input
from lib.CConv1D import CConv1D
from make_train_data import make_train_data
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

n_samples = 10**7
n_samples_mult_pairs = 10**6
bs = 5000
wdir = './models/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)


def make_checkpoint(file):
    return ModelCheckpoint(file, monitor='val_loss', save_best_only=True)


# make residual tower of convolutional blocks
def make_resnet(
        num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5, dilation_rate=1,reg_param=0.0001,
        final_activation='sigmoid', cconv=False
):
    Conv = CConv1D if cconv else SeparableConv1D  # Check if we use circular convolutions
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, dilation_rate=1,  padding='same',kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv(num_filters, kernel_size=ks, padding='same',  dilation_rate=2,kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv(num_filters, kernel_size=ks, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_param))(conv1)
        # conv2 = BatchNormalization()(conv2)
        # conv2 = Activation('tanh')(conv2)
        # conv3 = Conv(num_filters, kernel_size=ks, padding='same', dilation_rate=5, kernel_regularizer=l2(reg_param))(conv2)
        # conv3 = BatchNormalization()(conv3)
        # conv3 = Activation('relu')(conv3)
        shortcut = Add()([shortcut, conv2])
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    # dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    # dense2 = BatchNormalization()(dense2)
    # dense2 = Activation('tanh')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense1)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model


def train_distinguisher(
        cipher, diff, n_train_samples=10**7, n_val_samples=10**6, n_epochs=50, depth=10, n_neurons=64, kernel_size=3,
        n_filters=32, reg_param=10 ** -5, lr_high=0.002, lr_low=0.0001, cconv=False, calc_back=0):
    n_rounds = cipher.get_n_rounds()
    cipher_name = type(cipher).__name__
    result_base_name = f'{wdir}{cipher_name}_{n_rounds}_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # create the network
    net = make_resnet(
        depth=depth, d1=n_neurons, d2=n_neurons, ks=kernel_size, num_filters=n_filters, reg_param=reg_param,
        cconv=cconv, word_size=cipher.get_word_size(), num_blocks=cipher.get_n_words()
    )
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    X, Y = make_train_data(n_train_samples, cipher, diff, calc_back)
    X_eval, Y_eval = make_train_data(n_val_samples, cipher, diff, calc_back)
    # set up model checkpoint
    check = make_checkpoint(f'{result_base_name}.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, lr_high, lr_low))
    # train and evaluate
    h = net.fit(X, Y, epochs=n_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    np.save(wdir + 'h' + str(n_rounds) + 'r_depth' + str(depth) + '.npy', h.history['val_acc']);
    np.save(wdir + 'h' + str(n_rounds) + 'r_depth' + str(depth) + '.npy', h.history['val_loss']);
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    # np.save(f'{result_base_name}_h.npy', h.history['val_acc'])
    # np.save(f'{result_base_name}_h.npy', h.history['val_loss'])
    # #dump(h.history, open(f'{result_base_name}_hist.p', 'wb'))
    print(f'Best validation accuracy: {np.max(h.history["val_acc"])}, model saved as {result_base_name}.h5')

    # 显示训练集和验证集的acc和loss曲线
    # 训练集acc
    acc = h.history['acc']
    np.save("./acc_loss/7train_acc.npy", acc)
    # 测试集acc
    val_acc = h.history['val_acc']
    np.save("./acc_loss/7val_acc.npy", val_acc)
    # 训练集loss
    loss = h.history['loss']
    np.save("./acc_loss/7train_loss.npy", loss)
    # 测试集Loss
    val_loss = h.history['val_loss']
    np.save("./acc_loss/7val_loss.npy", val_loss)
    return net, h

if __name__ == '__main__':
    # fig=plt.figure(1)
    # axes = fig.subplots(nrows=1, ncols=2)
    fig=plt.figure(figsize=(12,10))
    acc=np.load("./acc_loss/7train_acc.npy")
    val_acc=np.load("./acc_loss/7val_acc.npy")
    loss=np.load("./acc_loss/7train_loss.npy")
    val_loss=np.load("./acc_loss/7val_loss.npy")
    # plt.ylim(0.6,0.96)
    # plt.grid(False)
    # plt.plot(loss, '-', linewidth=3.0, label='Training Set')
    # plt.plot(val_loss, '-', linewidth=3.0, label='Validation Set')
    #
    # # 设置坐标标签标注和字体大小
    # plt.xlabel('训练轮数（epoch)', fontsize=20)
    # plt.ylabel('准确率', fontsize=20)
    #
    #
    # # 设置坐标刻度字体大小
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # legend = plt.legend(loc='lower right', fontsize=20,  borderaxespad=1, ncol=1)
    # legend = plt.legend(loc='upper right',fontsize=20, bbox_to_anchor=(1, 1), borderaxespad=1, ncol=2)
    # title = plot.title(filename + '\n_' + str(nb_traces) + 'trs_' + str(nb_attacks) + 'att', loc='center')

    # plt.savefig('fig/Present_7acc.svg',
    #              format='svg', dpi=1200, bbox_extra_artists=(legend,), bbox_inches='tight')

    # loss=np.load("./7round/train_loss.npy")
    # val_loss=np.load("./7round/val_loss.npy")
    plt.rcParams['figure.figsize'] = (12, 10)
    # plt.ylim(0.6,1)
    plt.grid(False)

    plt.plot(loss, '-', linewidth=3.0, label='Training Set')
    plt.plot(val_loss, '-', linewidth=3.0, label='Validation Set')

    # 设置坐标标签标注和字体大小
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    ## 设置坐标刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    legend = plt.legend(loc='upper right', fontsize=20,bbox_to_anchor=(1, 1), borderaxespad=1, ncol=1)
    # title = plot.title(filename + '\n_' + str(nb_traces) + 'trs_' + str(nb_attacks) + 'att', loc='center')

    plt.savefig('fig/Present_7loss.pdf',
                 format='pdf', dpi=1200, bbox_extra_artists=(legend,), bbox_inches='tight')
