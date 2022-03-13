"""
@Time   :   2021-01-12 16:04:23
@File   :   dataset.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .utils import load_json
from src.pinyin_tool import PinyinTool
from transformers.models.bert.tokenization_bert import BertTokenizer


class NerDataset(Dataset):
    def __init__(self, fp, pretain_path):
        self.data = load_json(fp)
        self.space_char = "[unused1]"

        py_dict_path = './pinyin_data/zi_py.txt'
        py_vocab_path = './pinyin_data/py_vocab.txt'
        sk_dict_path = './stroke_data/zi_sk.txt'
        sk_vocab_path = './stroke_data/sk_vocab.txt'
        label2id_path = "data/label2ids.json"
        with open(label2id_path, 'r') as f:
            self.label2ids = json.load(f)

        self.number_tag = len(self.label2ids)
        self.tokenizer = BertTokenizer.from_pretrained(pretain_path)

        self.pytool = PinyinTool(
            py_dict_path=py_dict_path, py_vocab_path=py_vocab_path, py_or_sk='py')
        self.sktool = PinyinTool(
            py_dict_path=sk_dict_path, py_vocab_path=sk_vocab_path, py_or_sk='sk')

        self.pylen = self.pytool.PYLEN
        self.sklen = self.sktool.PYLEN

        self.PYID2SEQ = self.pytool.get_pyid2seq_matrix()
        self.SKID2SEQ = self.sktool.get_pyid2seq_matrix()

        self.tokenid_pyid = {}
        self.tokenid_skid = {}
        for key in self.tokenizer.vocab:
            self.tokenid_pyid[self.tokenizer.vocab[key]
                              ] = self.pytool.get_pinyin_id(key)
            self.tokenid_skid[self.tokenizer.vocab[key]
                              ] = self.sktool.get_pinyin_id(key)
        self.max_sen_len = 512

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        sample = data['sample']
        tokens = [t.split("\t")[0] for t in sample]
        labels = [t.split("\t")[1] for t in sample]
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]
            labels = labels[0:(self.max_sen_len - 2)]

        _tokens = []
        _labels = []
        _lmask = []
        pinyin_ids = []
        stroke_ids = []
        _tokens.append("[CLS]")
        _lmask.append(1)
        _labels.append(self.label2ids["O"])
        pinyin_ids.append(np.zeros(self.pylen))
        stroke_ids.append(np.zeros(self.sklen))
        for token, label in zip(tokens, labels):
            _tokens.append(token.lower())
            _labels.append(self.label2ids[label])
            _lmask.append(1)
            pyid = self.pytool.get_pinyin_id(token)
            pinyin_ids.append(self.PYID2SEQ[pyid, :])
            skid = self.sktool.get_pinyin_id(token)
            stroke_ids.append(self.SKID2SEQ[skid, :])
        _tokens.append("[SEP]")
        _labels.append(self.label2ids["O"])
        _lmask.append(1)
        pinyin_ids.append(np.zeros(self.pylen))
        stroke_ids.append(np.zeros(self.sklen))
        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)

        length = len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * length
        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_sen_len:
            input_ids.append(0)
            input_mask.append(0)
            pinyin_ids.append(np.zeros(self.pylen))
            stroke_ids.append(np.zeros(self.sklen))
            _labels.append(self.label2ids["O"])
            _lmask.append(0)
        pinyin_ids = np.vstack(pinyin_ids)
        stroke_ids = np.vstack(stroke_ids)
        return {"input_ids": input_ids, "length": length, "input_mask": input_mask, "pinyin_ids": pinyin_ids, "stroke_ids": stroke_ids, "labels": _labels, "lmask": _lmask}

    def get_zi_py_matrix(self):
        pysize = 430
        matrix = []
        for k in range(len(self.tokenizer.vocab)):
            matrix.append([0] * pysize)

        for key in self.tokenizer.vocab:
            tokenid = self.tokenizer.vocab[key]
            pyid = self.pytool.get_pinyin_id(key)
            matrix[tokenid][pyid] = 1.
        return np.asarray(matrix, dtype=np.float32)


def collate_fn(batches):
    input_ids = []
    input_masks = []
    pinyin_ids = []
    stroke_ids = []
    labels = []
    lmasks = []
    lengthes = []
    for batch in batches:
        input_ids.append(batch['input_ids'])
        input_masks.append(batch['input_mask'])
        pinyin_ids.append(batch['pinyin_ids'])
        stroke_ids.append(batch['stroke_ids'])
        labels.append(batch['labels'])
        lmasks.append(batch['lmask'])
        lengthes.append(batch['length'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.float32)
    pinyin_ids = torch.from_numpy(np.stack(pinyin_ids)).type(torch.long)
    stroke_ids = torch.from_numpy(np.stack(stroke_ids)).type(torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    lmasks = torch.tensor(lmasks, dtype=torch.float)
    lmasks = lmasks.type(torch.ByteTensor)
    return {"input_ids": input_ids, "length": lengthes, "input_mask": input_masks, "pinyin_ids": pinyin_ids, "stroke_ids": stroke_ids, "labels": labels, "lmask": lmasks}
