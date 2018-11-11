# coding=utf-8
"""
Author:     YangMing li
StartTime:  18/11/11
Filename:   main.py
Software:   Pycharm
LastModify: 18/11/12
"""

import os
import random
import numpy as np

import torch

from utils import loader
from utils import model
from utils import helper


H_PARAM_DATA_DIR = "./data"         # 数据文件的所在目录路径.
H_PARAM_TRAIN_BATCH = 32            # 训练时使用的 batch 大小.
H_PARAM_DEV_BATCH = 200             # 开发集评分时使用的 batch 大小.
H_PARAM_TEST_BATCH = 200            # 测试集评分时使用的 batch 大小.
H_PARAM_MAX_LETTER_LEN = 10         # 控制 padding 的最大长度.
H_PARAM_RANDOM_SEED = 0             # 训练模型时伪随机数种子.
H_PARAM_TRAINING_EPOCH = 100        # 训练时, 模型迭代数据集的轮数.
H_PARAM_LEARNING_RATE = 1e-3        # 训练时, 模型更新参数的步长.
H_PARAM_NORM_PENALTY = 1e-6         # 更新时, 2 阶范数惩罚的大小.

H_PARAM_KERNEL_SIZE = 4             # 做字符卷积时, 滑动核的宽度大小.
H_PARAM_WORD_EMBEDDING = 128        # 编码词向量的维度大小.
H_PARAM_LETTER_EMBEDDING = 12       # 编码字符向量的维度大小.
H_PARAM_ATTENTION_DIM = 64          # 自注意力机制的输出维度大小.
H_PARAM_DROPOUT_RATE = 0.2          # 网络 dropout 的概率值.
H_PARAM_LSTM_HIDDEN_DIM = 256       # LSTM 层迭代的隐层维度大小.

# 规定各个库函数的随机数种子.
torch.manual_seed(H_PARAM_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(H_PARAM_RANDOM_SEED)
    torch.cuda.manual_seed_all(H_PARAM_RANDOM_SEED)
random.seed(H_PARAM_RANDOM_SEED)
np.random.seed(H_PARAM_RANDOM_SEED)

# 读取训练, 测试过程所需的数据, 包括构造字符集.
train_file_path = os.path.join(H_PARAM_DATA_DIR, 'train.txt')  # 读取训练集 + 分解字符.
train_text_sent_list, train_text_slot_list, train_text_intent_list = loader.read_file(train_file_path)
train_text_letter_list = helper.iterative_support(helper.get_letter, train_text_sent_list)

dev_file_path = os.path.join(H_PARAM_DATA_DIR, 'dev.txt')  # 读取开发集 + 分解字符.
dev_text_sent_list, dev_text_slot_list, dev_text_intent_list = loader.read_file(dev_file_path)
dev_text_letter_list = helper.iterative_support(helper.get_letter, dev_text_sent_list)

test_file_path = os.path.join(H_PARAM_DATA_DIR, 'test.txt')  # 读取测试集 + 分解字符.
test_text_sent_list, test_text_slot_list, test_text_intent_list = loader.read_file(test_file_path)
test_text_letter_list = helper.iterative_support(helper.get_letter, test_text_sent_list)

# 实例化字典对象, 并向字典对象填充内容.
word_alphabet = loader.Alphabet(True, True)
letter_alphabet = loader.Alphabet(True, True)
slot_alphabet = loader.Alphabet(False, False)
intent_alphabet = loader.Alphabet(False, False)

helper.iterative_support(word_alphabet.add, train_text_sent_list)
helper.iterative_support(letter_alphabet.add, train_text_letter_list)
helper.iterative_support(slot_alphabet.add, train_text_slot_list)
helper.iterative_support(intent_alphabet.add, train_text_intent_list)

# 将某些文本类数据序列化, 但并非所有. 有下面两类:
#
#   1, 训练集的所有变量.
#   2, 开发集和测试集的 sent, char 集.
#
train_digit_sent_list = helper.iterative_support(word_alphabet.index, train_text_sent_list)
train_digit_letter_list = helper.iterative_support(letter_alphabet.index, train_text_letter_list)
train_digit_slot_list = helper.iterative_support(slot_alphabet.index, train_text_slot_list)
train_digit_intent_list = helper.iterative_support(intent_alphabet.index, train_text_intent_list)
dev_digit_sent_list = helper.iterative_support(word_alphabet.index, dev_text_sent_list)
dev_digit_letter_list = helper.iterative_support(letter_alphabet.index, dev_text_letter_list)
test_digit_sent_list = helper.iterative_support(word_alphabet.index, test_text_sent_list)
test_digit_letter_list = helper.iterative_support(letter_alphabet.index, test_text_letter_list)

# 构建 train, dev, test 等的 batch 分发对象.
train_loader = loader.batch_delivery(
    train_digit_sent_list, train_digit_letter_list,
    train_digit_slot_list, train_digit_intent_list,
    H_PARAM_TRAIN_BATCH, True
)
dev_loader = loader.batch_delivery(
    dev_digit_sent_list, dev_digit_letter_list,
    dev_text_slot_list, dev_text_intent_list,
    H_PARAM_DEV_BATCH, False
)
test_loader = loader.batch_delivery(
    test_digit_sent_list, test_digit_letter_list,
    test_text_slot_list, test_text_intent_list,
    H_PARAM_TEST_BATCH, False
)

model = model.GatedAttentionSLU(
    len(word_alphabet), len(letter_alphabet),
    len(slot_alphabet), len(intent_alphabet),
    H_PARAM_WORD_EMBEDDING, H_PARAM_LETTER_EMBEDDING,
    H_PARAM_ATTENTION_DIM, H_PARAM_LSTM_HIDDEN_DIM,
    H_PARAM_KERNEL_SIZE, H_PARAM_DROPOUT_RATE
)

process = helper.Process(
    model, H_PARAM_TRAINING_EPOCH, H_PARAM_LEARNING_RATE,
    H_PARAM_NORM_PENALTY, H_PARAM_MAX_LETTER_LEN, train_loader,
    dev_loader, test_loader, slot_alphabet, intent_alphabet
)
process.train_model()
