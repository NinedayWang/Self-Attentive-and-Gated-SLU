# coding=utf-8
"""
Author:     YangMing li
StartTime:  18/11/11
Filename:   utils/helper.py
Software:   Pycharm
LastModify: 18/11/12
"""

import time
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils.miulab import computeF1Score


def iterative_support(func, x):
    """
    支持某个函数 func 对输入 x 的迭代查询.
    """

    if isinstance(x, (list, tuple, np.ndarray)):
        return [iterative_support(func, elem) for elem in x]

    # 预检测, 防 bug.
    assert isinstance(x, (int, str, unicode))

    return func(x)


def get_letter(word):
    """
    将字符串拆开, 返回其组成字符的列表.
    """

    return list(word)


def expand_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item


def nest_list(items, seq_lens):
    num_items = len(items)
    trans_items = [[] for _ in range(0, num_items)]

    count = 0
    for jdx in range(0, len(seq_lens)):
        for idx in range(0, num_items):
            trans_items[idx].append(items[idx][count:count+seq_lens[jdx]])
        count += seq_lens[jdx]

    return trans_items


class Padding(object):
    """
    这里均假设是对一个 batch(多个句子的列表) 做 padding. 需要
    注意的是 sent 要 padding 要排序, label 只要排序.
    """

    @staticmethod
    def word_padding(sent_list, letter_list, items_list):
        """
        对句子做 padding, 填充 0.

        :param sent_list: 是句子(词表)的列表.
        :param letter_list: 是字符级的列表.
        :param items_list: 是其他标注列表的列表.
        :return: 返回排序且 padding 后的句表, 排序后
                 的标注列表, 和句子长度表 seq_lens.
        """

        seq_lens = [len(element) for element in sent_list]
        max_len = max(seq_lens)

        # 对句子表进行排序, 然后将 index 的表存下来.
        arg_list = np.argsort(seq_lens)[::-1]

        m_sent_list, m_letter_list, m_seq_lens = [], [], []
        m_items_list = [[] for _ in range(0, len(items_list))]

        for index in arg_list:
            m_seq_lens.append(seq_lens[index])
            m_sent_list.append(deepcopy(sent_list[index]))  # 用 copy 防止污染对象.
            m_sent_list[-1].extend([0] * (max_len - seq_lens[index]))

            m_letter_list.append(deepcopy(letter_list[index]))
            m_letter_list[-1].extend([[] for _ in range(0, max_len - seq_lens[index])])

            for item_i in range(0, len(items_list)):
                m_items_list[item_i].append(items_list[item_i][index])

        return m_sent_list, m_letter_list, m_items_list, m_seq_lens

    @staticmethod
    def letter_padding(letter_list, bound_len):
        max_len = 0
        for sent in letter_list:
            for word in sent:
                max_len = max(len(word), max_len)

        m_letter_list = []
        for sent_i in range(0, len(letter_list)):
            m_letter_list.append([])
            for word_i in range(0, len(letter_list[sent_i])):
                word = letter_list[sent_i][word_i]
                m_word = word + [0] * (max_len - len(word))
                m_letter_list[-1].append(m_word[:bound_len])

        return m_letter_list


class Metric(object):
    """
    度量类, 提供计算 acc 和 f1 的静态方法.
    """

    @staticmethod
    def get_acc(pred_list, real_list):
        pred_list = list(expand_list(pred_list))
        real_list = list(expand_list(real_list))

        # 预检测, 防 bug.
        assert len(pred_list) == len(real_list)

        correct, total = 0.0, 0.0
        for pred, real in zip(pred_list, real_list):
            if pred == real:
                correct += 1.0
            total += 1.0

        return 1.0 * correct / total

    @staticmethod
    def get_f1(pred_list, real_list):
        return computeF1Score(real_list, pred_list)[0]


class Process(object):
    """
    提供了用于训练, 测试 nn.Module 的方法.
    """

    def __init__(self, model, num_epoch, learning_rate, l2_penalty,  max_letter_len,
                 train_loader, dev_loader, test_loader, slot_alphabet, intent_alphabet):

        self._model = model
        self._num_epoch = num_epoch
        self._max_letter_len = max_letter_len
        self._slot_alphabet = slot_alphabet
        self._intent_alphabet = intent_alphabet

        if torch.cuda.is_available():
            self._model = self._model.cuda()

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=l2_penalty
        )
        self._criterion = nn.NLLLoss()

        self._train_loader = train_loader
        self._dev_loader = dev_loader
        self._test_loader = test_loader

    def train_model(self):
        self._model.train()

        best_dev_slot_slot, best_dev_intent_intent = 0.0, 0.0
        best_dev_both_slot, best_dev_both_intent = 0.0, 0.0

        test_slot_slot, test_slot_intent = 0.0, 0.0
        test_intent_slot, test_intent_intent = 0.0, 0.0
        test_both_slot, test_both_intent = 0.0, 0.0

        for epoch in range(0, self._num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0

            epoch_time_start = time.time()
            for sent, letter, slot, intent in self._train_loader:
                p_sent, p_letter, [p_slot, p_intent], seq_lens = Padding.word_padding(
                    sent, letter, [slot, intent]
                )
                e_letter = Padding.letter_padding(p_letter, self._max_letter_len)

                e_slot = list(expand_list(p_slot))
                e_intent = list(expand_list(p_intent))

                var_sent = Variable(torch.LongTensor(p_sent))
                var_letter = Variable(torch.LongTensor(e_letter))
                var_slot = Variable(torch.LongTensor(e_slot))
                var_intent = Variable(torch.LongTensor(e_intent))

                if torch.cuda.is_available():   # 支持 CUDA 时, 用 GPU 加快运算.
                    var_sent = var_sent.cuda()
                    var_letter = var_letter.cuda()
                    var_slot = var_slot.cuda()
                    var_intent = var_intent.cuda()

                pred_slot, pred_intent = self._model(var_sent, var_letter, seq_lens)

                slot_loss = self._criterion(pred_slot, var_slot)
                intent_loss = self._criterion(pred_intent, var_intent)
                total_loss = slot_loss + intent_loss
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                total_slot_loss += slot_loss.cpu().data.numpy()[0]
                total_intent_loss += intent_loss.cpu().data.numpy()[0]

            epoch_time_con = time.time() - epoch_time_start
            print "[轮数 {:03d}]: 模型在 slot 上的损失为: {:04.6f}, 在 intent 上的损失为: {:04.6f}, 共耗" \
                  "时: {:03.6f} 秒.\n".format(epoch, total_slot_loss, total_intent_loss, epoch_time_con)

            dev_slot, dev_intent = self._estimate(True)  # 检查模型在 dev 集的效果.

            slot_flag = dev_slot >= best_dev_slot_slot
            intent_flag = dev_intent >= best_dev_intent_intent
            both_flag = dev_slot >= best_dev_both_slot and dev_intent >= best_dev_both_intent

            time_estimate_start = time.time()
            if slot_flag or intent_flag or both_flag:
                test_slot, test_intent = self._estimate(False)

                if slot_flag:
                    test_slot_slot = test_slot
                    test_slot_intent = test_intent

                if intent_flag:
                    test_intent_slot = test_slot
                    test_intent_intent = test_intent

                if both_flag:
                    test_both_slot = test_slot
                    test_both_intent = test_intent

                print "[优先 -slot-]: 模型在 test 集中, slot f1: {:4.6f}, 且 intent " \
                      "acc: {:4.6f};".format(test_slot_slot, test_slot_intent)
                print "[优先 intent]: 模型在 test 集中, slot f1: {:4.6f}, 且 intent " \
                      "acc: {:4.6f};".format(test_intent_slot, test_intent_intent)
                print "[优先 -both-]: 模型在 test 集中, slot f1: {:4.6f}, 且 intent " \
                      "acc: {:4.6f};\n".format(test_both_slot, test_both_intent)

            time_estimate_con = time.time() - time_estimate_start
            print "[轮数 {:03d}]: 模型在 dev 集上, slot f1 为 {:4.6f} , 且 intent acc 为 {:4.6f}, 测" \
                  "试时间开销 {:3.6f} 秒.".format(epoch, dev_slot, dev_intent, time_estimate_con)

    def _estimate(self, dev_mode):
        self._model.eval()

        pred_slot_list, real_slot_list = [], []
        pred_intent_list, real_intent_list = [], []

        loader = self._dev_loader if dev_mode else self._test_loader
        for sent, letter, slot, intent in loader:
            p_sent, p_letter, [p_slot, p_intent], seq_lens = Padding.word_padding(
                sent, letter, [slot, intent]
            )
            e_letter = Padding.letter_padding(p_letter, self._max_letter_len)

            real_slot_list.extend(p_slot)
            real_intent_list.extend(p_intent)

            var_sent = Variable(torch.LongTensor(p_sent))
            var_letter = Variable(torch.LongTensor(e_letter))
            if torch.cuda.is_available():
                var_sent = var_sent.cuda()
                var_letter = var_letter.cuda()

            pred_slot, pred_intent = self._model(var_sent, var_letter, seq_lens)

            _, slot_index = pred_slot.topk(1, dim=1)
            slot_index = list(expand_list(slot_index.cpu().data.numpy().tolist()))
            slot_index = nest_list([slot_index], seq_lens)[0]
            pred_slot_list.extend(iterative_support(self._slot_alphabet.get, slot_index))

            _, intent_index = pred_intent.topk(1, dim=1)
            intent_index = intent_index.cpu().data.numpy().tolist()
            intent_index = list(expand_list(intent_index))
            pred_intent_list.extend(iterative_support(self._intent_alphabet.get, intent_index))

        slot_f1 = Metric.get_f1(pred_slot_list, real_slot_list)
        intent_acc = Metric.get_acc(pred_intent_list, real_intent_list)

        return slot_f1, intent_acc
