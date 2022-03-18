# -*- encoding: utf-8 -*-
'''
File    :   convert_tf_to_torch.py
Time    :   2022/03/15 21:08:46
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   将TensorFlow模型转换为torch模型
'''
from ast import comprehension
from transformers import convert_tf_weight_name_to_pt_weight_name
from collections import OrderedDict, defaultdict
import numpy as np
import os
import re
import argparse
import deepdish as dd
import tfpyth
import torch
import tensorflow as tf
pretrain_path = "/mnt/disk2/PythonProgram/NLPCode/PretrainModel/PLOME/pretrained_plome"


# session = tf.Session()


def get_torch_function():
    input_ids = tf.placeholder(tf.float32, name="input_ids")
    input_mask = tf.placeholder(tf.float32, name='input_mask')
    pinyin_ids = tf.placeholder(tf.float32, name="pinyin_ids")
    lmask = tf.placeholder(tf.float32, name='lmask')
    label_ids = tf.placeholder(tf.float32, name="label_ids")
    masked_pinyin_ids = tf.placeholder(tf.float32, name='masked_pinyin_ids')


def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (
        n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights


def convert_tf_to_torch(input_file, output_file):
    """_summary_

    Args:
        input_file (_type_): _description_
        output_file (_type_): _description_
    """
    weights = read_ckpt(input_file)
    dd.io.save(output_file, weights, compression=None)
    weights2 = dd.io.load(output_file)
    new_pre_dict = OrderedDict()
    for k, v in weights2.items():
        # 将参数值numpy转换为tensor形式
        new_pre_dict[k] = torch.tensor(v)
    torch.save(new_pre_dict, "./data/plome.pt")


def convert_gru_model(weights_static, name):
    """将tf的gru权重转换成torch的gru权重
        1. 由于tf的gru模型是先合并计算重置门和更新门(2*hidden_size)，即将输入和隐藏层进行concat(hidden_size)，作为输入,(这时的权重为:(2*hidden_size,input_size+hidden_size)),注意input_size 在前；
            然后再更新隐藏层(hidden_size),即将更新门和隐藏层的乘积，与输入进行concat后(input_size+hidden_size)，作为输入(这时的权重为:(hidden_size,input_size+hidden_size)),注意input_size 在前。
        2. torch是分别用权重去和隐藏层、输入相乘积，那么该权重分别为(3*hidden_size,hidden_size)、(3*hidden_size,input_size)。

        现在要做的是将tf的gru权重移植到torch的gru权重中
    """
    torch_weights = OrderedDict()
    weight_template_name = "weight_{}h_l{}"
    bias_template_name = "bias_{}h_l{}"
    new_weightes = defaultdict(dict)
    for k, v in weights_static.items():
        k = k.split(".")[-2:]
        if len(k) == 1:
            new_weightes[k[0]] = v
        if len(k) == 2:
            new_weightes[k[0]][k[1]] = v
    candidate_weight = new_weightes['candidate']['weight']
    hidden_size, hidden_input_size = candidate_weight.shape
    gates_weight = new_weightes['gates']['weight']
    torch_gru_hidden_weight = torch.concat(
        (gates_weight[:, -hidden_size:], candidate_weight[:, -hidden_size:]))
    torch_gru_input_weight = torch.concat(
        (gates_weight[:, :-hidden_size], candidate_weight[:, :-hidden_size]))
    torch_gru_bias = torch.concat(
        (new_weightes['gates']['bias'], new_weightes['candidate']['bias']))
    for k, v in new_weightes.items():
        if isinstance(k, str) and "_emb" in k:
            torch_embedding = v.transpose(1, 0)
    torch_weights[name+".pyemb.weight"] = torch_embedding
    torch_weights[name+".GRU." +
                  weight_template_name.format("i", 0)] = torch_gru_input_weight
    torch_weights[name+".GRU." +
                  weight_template_name.format("h", 0)] = torch_gru_hidden_weight
    torch_weights[name+".GRU." +
                  bias_template_name.format("i", 0)] = torch_gru_bias
    torch_weights[name+".GRU." +
                  bias_template_name.format("h", 0)] = torch_gru_bias
    return torch_weights


def convert_bert_model(bert_static):
    """_summary_

    Args:
        bert_static (_type_): _description_
    """
    layer_number_compile = re.compile("_(\d+)")
    bert_embeddings = {}
    bert_encoder = {}
    for k1, v in bert_static.items():
        k = k1[5:]
        if "embeddings" in k:
            if not k.endswith("weight") and not k.endswith("bias"):
                k = k+".weight"
                v = v.transpose(1, 0)
            # bert_embeddings[k] = v
            bert_encoder[k] = v
        if k.startswith("encoder"):
            layernumer = layer_number_compile.search(k).group(0)
            k = layer_number_compile.sub(r".\1", k)
            layernumer = int(layernumer[1:])
            # bert_encoder[layernumer].append(v)
            bert_encoder[k] = v

    return bert_encoder


def convert_tf_weight_to_torch_weight():
    """_summary_
    """
    model = torch.load("./data/plome.pt")
    self_model = torch.load("./data/init.pt")
    sk_emb = {}
    py_emb = {}
    bert_ = {}
    for k, v in model.items():
        # 这是由于采用的是adam优化器，tf框架在保存模型的时候，通常会将每个参数的一阶矩(adam_m)和二阶矩(adam_v)保存下来，
        # 但是这些参数在预测的时候，往往是不需要的。
        if "adam_v" in k or "adam_m" in k:
            continue
        # print(k)
        # 这是sk embedding相关
        if "sk_emb" in k:
            sk_emb[convert_tf_weight_name_to_pt_weight_name(k)[0]] = v

        if "py_emb" in k:
            py_emb[convert_tf_weight_name_to_pt_weight_name(k)[0]] = v

        if "bert" in k:
            bert_[convert_tf_weight_name_to_pt_weight_name(k)[0]] = v
    plome_static_weight = OrderedDict()
    sk_emb_weight = convert_gru_model(sk_emb, "sk_emb")
    py_emb_weight = convert_gru_model(py_emb, "py_emb")
    bert_encoder = convert_bert_model(bert_)
    plome_static_weight.update(sk_emb_weight)
    plome_static_weight.update(py_emb_weight)
    plome_static_weight.update(bert_encoder)

    self_model_static = OrderedDict()
    for k, v in plome_static_weight.items():
        if (v.shape != self_model[k].shape):
            print(k, v.shape, self_model[k].shape)
        self_model_static[k] = v

    torch.save(self_model_static, './data/new_init.pt')


if __name__ == '__main__':
    convert_tf_to_torch(
        "/mnt/disk2/PythonProgram/NLPCode/PretrainModel/PLOME/pretrained_plome/bert_model.ckpt", './data/plome.h5')
    # convert_tf_weight_to_torch_weight()
    pass
