# coding=utf-8
"""
Author:     YangMing li
StartTime:  18/11/11
Filename:   utils/loader.py
Software:   Pycharm
LastModify: 18/11/11
"""

import codecs
from collections import Counter

from torch.utils.data import Dataset, DataLoader


SIGN_PAD = "<PAD>"  # Padding 的字符串标识.
SIGN_UNK = "<UNK>"  # 未知词 UNK 的字符串标识.


class Alphabet(object):
    """
    字典类, 对某个有限集合的元素提供存储和序列化操
    作.对陌生词的序列化, API 提供两种解决方案:

        1, 设置一个陌生词标记 "<UNK>".
        2, 记录频率最高的词, 并作为预测.

    推荐对词集使用第一种, 对标记集使用第二种.

    另外, 由于 NLP 要对边长输入做处理, 即 padding,
    对于词集, 我们新增 "<PAD>" 这个符号, 并标为 0.
    """

    def __init__(self, use_unk, use_pad):

        self.index2instance = []  # 符号化: 序号 -> 元素.
        self.instance2index = {}  # 序列化: 元素 -> 序号.

        self.freq_counter = Counter()  # 记录对原始文本统计的频率.

        if use_pad:
            self.add(SIGN_PAD)

        self.use_unk = use_unk  # 使用未知词标识符.
        if use_unk:
            self.add(SIGN_UNK)

    def add(self, instance):
        """
        向字典对象中增加一个元素.
        """

        self.freq_counter[instance] += 1

        if instance not in self.index2instance:
            self.instance2index[instance] = len(self.index2instance)
            self.index2instance.append(instance)

    def index(self, instance):
        """
        查询某个元素的序列号. 根据参数 use_unk 对
        未知词使用最大频预测或是标识符预测.
        """

        try:
            return self.instance2index[instance]
        except KeyError:
            if self.use_unk:
                return self.instance2index[SIGN_UNK]
            else:
                item = self.freq_counter.most_common(1)[0][0]
                return self.instance2index[item]

    def get(self, index):
        """
        获取某个序号 index 对应在字典中的元素, 它
        不允许有错误, 越界或类型错误就直接报错.
        """

        return self.index2instance[index]

    def __len__(self):
        return len(self.instance2index)


def read_file(file_path):
    """
    读取给定路径下的数据文件.
    """

    sent_list, slot_list, intent_list = [], [], []
    with codecs.open(file_path, 'r', encoding='utf-8') as fr:
        sent, slot = [], []  # 设置缓存段存储读入.

        for line in fr.readlines():
            items = line.strip().split()

            if len(items) == 2:
                sent.append(items[0].strip())
                slot.append(items[1].strip())
            elif len(items) == 1:
                sent_list.append(sent)
                slot_list.append(slot)
                intent_list.append(items)

                # 清空缓存, 等待下一个输入样例.
                sent, slot = [], []

    return sent_list, slot_list, intent_list


class DataManager(Dataset):

    def __init__(self, sent, letter, slot, intent):
        self.sent = sent
        self.letter = letter
        self.slot = slot
        self.intent = intent

    def __getitem__(self, item):
        return self.sent[item], self.letter[item], self.slot[item], self.intent[item]

    def __len__(self):
        return len(self.sent)


def collate_func(batch):
    """
    作为实例化 DataLoader 对象的函数参数.
    """

    n_entity = len(batch[0])
    r_batch = [[] for _ in range(0, n_entity)]

    for idx in range(0, len(batch)):
        for jdx in range(0, n_entity):
            r_batch[jdx].append(batch[idx][jdx])

    return r_batch


def batch_delivery(sent_list, letter_list, slot_list, intent_list, batch_size, shuffle):
    """
    返回一个 DataLoad 的对象, 用于对数据集做 batch.
    """

    torch_data = DataManager(
        sent_list, letter_list,
        slot_list, intent_list
    )
    return DataLoader(
        torch_data, batch_size=batch_size,
        shuffle=shuffle, collate_fn=collate_func
    )
