# coding=utf-8
"""
Author:     YangMing li
StartTime:  18/11/11
Filename:   utils/model.py
Software:   Pycharm
LastModify: 18/11/11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionSLU(nn.Module):

    def __init__(self, word_vocab_num, letter_vocab_num, slot_vocab_num, intent_vocab_num,
                 word_embedding_dim, letter_embedding_dim, self_attention_dim, lstm_hidden_dim,
                 kernel_size, dropout_rate):
        super(GatedAttentionSLU, self).__init__()

        self._embedding_layer = Embedding(
            word_vocab_num, letter_vocab_num,
            word_embedding_dim, letter_embedding_dim,
            kernel_size, dropout_rate
        )
        self._e_attention = SelfAttention(
            word_embedding_dim, self_attention_dim, dropout_rate
        )
        self._lstm_layer = nn.LSTM(
            input_size=word_embedding_dim+self_attention_dim,
            hidden_size=lstm_hidden_dim // 2,   # 双向
            batch_first=True,
            num_layers=1,
            bidirectional=True,
            dropout=dropout_rate
        )
        self._d_attention = SelfAttention(
            lstm_hidden_dim, self_attention_dim, dropout_rate
        )
        self._intent_pred_linear = nn.Linear(
            lstm_hidden_dim, intent_vocab_num
        )
        self._intent_gate_linear = nn.Linear(
            self_attention_dim + intent_vocab_num, lstm_hidden_dim
        )
        self._slot_linear = nn.Linear(
            lstm_hidden_dim, slot_vocab_num
        )

    def forward(self, input_w, input_l, seq_lens):
        embedding_x = self._embedding_layer(input_w, input_l)
        attention_x = self._e_attention(embedding_x)
        emb_attn_x = torch.cat([embedding_x, attention_x], dim=-1)
        lstm_hidden, _ = self._lstm_layer(emb_attn_x)

        pool_hidden = torch.mean(lstm_hidden, dim=1, keepdim=True)
        linear_intent = self._intent_pred_linear(pool_hidden)

        # 预测一个 batch 的 intent 负对数分布.
        pred_intent = F.log_softmax(linear_intent.squeeze(1), dim=-1)

        rep_intent = torch.cat([linear_intent] * max(seq_lens), dim=1)
        attn_hidden = self._d_attention(lstm_hidden)
        com_hidden = torch.cat([rep_intent, attn_hidden], dim=-1)
        lin_hidden = self._intent_gate_linear(com_hidden)
        gated_hidden = lin_hidden * lstm_hidden

        linear_slot = self._slot_linear(gated_hidden)
        expand_slot = [linear_slot[i][:seq_lens[i], :] for i in range(0, len(seq_lens))]   # 去掉padding部分
        pred_slot = F.log_softmax(torch.cat(expand_slot, dim=0), dim=-1)

        return pred_slot, pred_intent


class Embedding(nn.Module):
    """
    嵌入向量模块, 包括词向量和字母向量. 其中
    后者要通过卷积提取特征.
    """

    def __init__(self, word_vocab_num, letter_vocab_num,
                 word_embedding_dim, letter_embedding_dim,
                 kernel_size, dropout_rate):

        super(Embedding, self).__init__()

        self._word_embedding_layer = nn.Embedding(word_vocab_num, word_embedding_dim // 2)
        self._letter_embedding_layer = nn.Embedding(letter_vocab_num, letter_embedding_dim)
        self._dropout_layer = nn.Dropout(dropout_rate)
        self._convolution_layer = []
        for i in range(len(kernel_size)):
            self._convolution_layer.append(nn.Conv1d(
                in_channels=letter_embedding_dim,
                out_channels=word_embedding_dim // (2*len(kernel_size)),      # 卷积核输出层的神经元个数（有多少个卷积核）
                kernel_size=kernel_size[i]      # 卷积核的宽度
            ))

    def forward(self, input_w, input_l):
        embedding_w = self._word_embedding_layer(input_w)

        l_size = input_l.size()
        embedding_l = self._letter_embedding_layer(input_l.view(-1, l_size[-1]))

        dropout_w = self._dropout_layer(embedding_w)
        dropout_l = self._dropout_layer(
            embedding_l.view(l_size[0] * l_size[1], l_size[2], -1)
        )

        pool_l_all = []
        for i in range(len(self._convolution_layer)):
            conv_l = self._convolution_layer[i](dropout_l.transpose(2, 1))
            pool_l = torch.mean(conv_l, dim=-1).view(l_size[0], l_size[1], -1)
            pool_l_all.append(pool_l)
        pool_l_all = torch.cat([pool_l_all[i] for i in range(len(pool_l_all))], dim=-1)

        combine_vec = torch.cat([dropout_w, pool_l_all], dim=2)
        return combine_vec


class SelfAttention(nn.Module):
    """
    基于 KVQ 计算模式的自注意力机制.
    """

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        self._k_matrix = nn.Linear(input_dim, output_dim)
        self._v_matrix = nn.Linear(input_dim, output_dim)
        self._q_matrix = nn.Linear(input_dim, output_dim)
        self._dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_x):
        k_x = self._k_matrix(input_x)
        v_x = self._v_matrix(input_x)
        q_x = self._q_matrix(input_x)

        drop_kx = self._dropout_layer(k_x)
        drop_vx = self._dropout_layer(v_x)
        drop_qx = self._dropout_layer(q_x)

        alpha = F.softmax(torch.matmul(drop_qx, drop_kx.transpose(-2, -1)), dim=-1)
        return torch.matmul(alpha, drop_vx)
