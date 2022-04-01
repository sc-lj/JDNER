# -*- encoding: utf-8 -*-
'''
File    :   searchSimWord.py
Time    :   2022/03/21 22:09:06
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   尝试通过训练好的ner模型，查询相似的词汇
'''
import json
import torch
import pytorch_lightning as pl
import os
import numpy as np
from collections import defaultdict
from modelsCRF import JDNerModel, BertModel, CRFNerTrainingModel
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
from main import parse_args
args = parse_args()
args.number_tag = 81
args.pylen, args.sklen = 6, 10


def tranf():
    path = "lightning_logs/version_1/checkpoints/epoch=62-f1=0.7933-pre=0.781-recall=0.806.ckpt"
    args.number_tag = 105
    model = CRFNerTrainingModel(args)

    model = model.load_from_checkpoint(path,  arguments=args)

    bert = model.model.bert
    torch.save(bert, "./data/ner_bert_state.pt")


def build_entity(tags):
    """构建实体span

    Args:
        tags (_type_): _description_
    """
    entities = []
    tmp_ent = ""
    start = 0
    for index, tag in enumerate(tags):
        if tag.startswith("B"):
            if tmp_ent:
                entities.append((tmp_ent, start, index))
            start = index
            tmp_ent = tag[2:]
        elif tag.startswith("O"):
            if tmp_ent and tmp_ent != "O":
                entities.append((tmp_ent, start, index))
                start = index
            tmp_ent = "O"
    if tmp_ent:
        entities.append((tmp_ent, start, index))
        tmp_ent = ""

    return entities


def search_text():
    device = torch.device("cuda")
    bert_model = torch.load("./data/ner_bert_state.pt")
    bert_model.eval()
    bert_model.to(device)
    with open("data/train.json", 'r') as f:
        data = json.load(f)
    tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint)
    max_sen_len = 512
    batch_input = defaultdict(dict)
    for line in tqdm(data):
        text = line['text']
        sample = line['sample']
        tokens = [t.split("\t")[0] for t in sample]
        labels = [t.split("\t")[1] for t in sample]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_sen_len - 2:
            tokens = tokens[0:(max_sen_len - 2)]
            labels = labels[0:(max_sen_len - 2)]

        _tokens = []
        _labels = []
        _tokens.append("[CLS]")
        _labels.append("O")
        for token, label in zip(tokens, labels):
            _tokens.append(token.lower())
            _labels.append(label)
        _tokens.append("[SEP]")
        _labels.append("O")

        label_index = build_entity(_labels)
        input_ids = tokenizer.convert_tokens_to_ids(_tokens)

        length = len(input_ids)
        input_mask = [1] * length
        while len(input_ids) < max_sen_len:
            input_ids.append(0)
            input_mask.append(0)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_ids = input_ids.to(device)
        input_mask = torch.tensor([input_mask], dtype=torch.float)
        input_mask = input_mask.to(device)
        with torch.no_grad():
            sequence_out = bert_model(
                input_ids=input_ids, attention_mask=input_mask)
        sequence_out = sequence_out.cpu()
        for line in label_index[1:-1]:
            label, s, e = line
            label_vec = sequence_out[0][s:e, :]
            label_vec = label_vec.mean(0).detach().numpy()
            word = "".join(_tokens[s:e])
            if word in batch_input[text]:
                root_vec = batch_input[text][word]
                label_vec = np.vstack([root_vec, label_vec])
                label_vec = label_vec.mean(0)
            batch_input[text][word] = label_vec

    with open("./data/entities_vec_sim.pkl", 'wb') as f:
        pickle.dump(batch_input, f)


if __name__ == "__main__":
    # tranf()
    search_text()
