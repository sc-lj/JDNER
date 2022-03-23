"""
@Time   :   2021-01-12 16:04:23
@File   :   dataset.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from ast import arg
from secrets import choice
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import load_json
from pinyin_tool import PinyinTool
from transformers.models.bert.tokenization_bert import BertTokenizer
import random
import synonyms


class NerDataset(Dataset):
    def __init__(self, fp, args, is_train=True):
        super().__init__()
        self.data = load_json(fp)
        self.space_char = "[unused1]"

        py_dict_path = './pinyin_data/zi_py.txt'
        py_vocab_path = './pinyin_data/py_vocab.txt'
        sk_dict_path = './stroke_data/zi_sk.txt'
        sk_vocab_path = './stroke_data/sk_vocab.txt'
        label2id_path = args.label_file
        with open(label2id_path, 'r') as f:
            self.label2ids = json.load(f)

        entity_path = args.entity_path
        with open(entity_path, 'r') as f:
            self.entities = json.load(f)

        self.is_train = is_train

        self.number_tag = len(self.label2ids)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint)

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
        if self.is_train and random.random() < 0.5:
            tokens, labels = self.EDA(tokens, labels)

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
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * length
        # Zero-pad up to the sequence length.
        # while len(input_ids) < self.max_sen_len:
        #     input_ids.append(0)
        #     input_mask.append(0)
        #     pinyin_ids.append(np.zeros(self.pylen))
        #     stroke_ids.append(np.zeros(self.sklen))
        #     _labels.append(self.label2ids["O"])
        #     _lmask.append(0)
        pinyin_ids = np.vstack(pinyin_ids)
        stroke_ids = np.vstack(stroke_ids)
        return {"input_ids": input_ids, "length": length, "input_mask": input_mask, "pinyin_ids": pinyin_ids, "stroke_ids": stroke_ids,
                "labels": _labels, "lmask": _lmask, "pylen": self.pylen, "sklen": self.sklen, "labelOid": self.label2ids["O"]}

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

    def EDA(self, tokens, labels):
        """实体识别的数据增强技术

        Args:
            tokens (_type_): _description_
            labels (_type_): _description_
        """
        entities = []
        tmp_ent = ""
        start = 0
        for index, tag in enumerate(labels):
            if tag.startswith("B"):
                if tmp_ent:
                    entities.append((tmp_ent, tokens[start:index]))
                start = index
                tmp_ent = tag[2:]
            elif tag.startswith("O"):
                if tmp_ent and tmp_ent != "O":
                    entities.append((tmp_ent, tokens[start:index]))
                    start = index
                tmp_ent = "O"
        if tmp_ent:
            entities.append((tmp_ent, tokens[start:index]))
            tmp_ent = ""
        new_tokens = []
        new_labels = []
        number = len(entities)
        is_choiced = False
        for i in range(number):
            line = entities[i]
            lab, token = line
            if lab == "O":
                new_tokens.extend(token)
                new_labels.extend(["O"]*len(token))
                continue
            elif not is_choiced:
                prob = random.random()
                if prob < 0.3:
                    # lab_entites = self.entities[lab]
                    # choice_ent = random.choice(lab_entites)
                    # 采用同义词查找
                    choice_ent = synonyms.nearby("".join(token))
                    choice_ent = [word for word, pro in zip(
                        *choice_ent) if 1 > pro > 0.66]
                    if len(choice_ent) != 0:
                        choice_ent = random.choice(choice_ent)
                        is_choiced = True
                        choice_ent_list = list(choice_ent)
                        choice_ent_list = [
                            l if len(l.strip()) else self.space_char for l in choice_ent_list]
                    else:
                        choice_ent_list = token

                    new_tokens.extend(choice_ent_list)
                    new_labels.extend(
                        ["B-"+lab if i == 0 else "I-"+lab for i in range(len(choice_ent_list))])
                else:
                    new_tokens.extend(token)
                    new_labels.extend(
                        ["B-"+lab if i == 0 else "I-"+lab for i in range(len(token))])
            else:
                new_tokens.extend(token)
                new_labels.extend(
                    ["B-"+lab if i == 0 else "I-"+lab for i in range(len(token))])
        return new_tokens, new_labels


def collate_fn(batches):
    max_length = max([batch['length'] for batch in batches])
    input_ids = []
    input_masks = []
    pinyin_ids = []
    stroke_ids = []
    labels = []
    lmasks = []
    lengthes = []
    for batch in batches:
        length = batch['length']
        pylen = batch["pylen"]
        sklen = batch['sklen']
        labelOid = batch["labelOid"]
        input_ids.append(batch['input_ids']+[0]*(max_length-length))
        input_masks.append(batch['input_mask']+[0]*(max_length-length))
        pinyin_id = batch['pinyin_ids'] + \
            [np.zeros((max_length-length), pylen)]
        pinyin_id = np.vstack(pinyin_id)
        pinyin_ids.append(pinyin_id)
        stroke_id = batch['stroke_ids'] + \
            [np.zeros((max_length-length), sklen)]
        stroke_id = np.vstack(stroke_id)
        stroke_ids.append(stroke_id)
        labels.append(batch['labels']+[labelOid]*(max_length-length))
        lmasks.append(batch['lmask']+[0]*(max_length-length))
        lengthes.append(batch['length'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.float32)
    pinyin_ids = torch.from_numpy(np.stack(pinyin_ids)).type(torch.long)
    stroke_ids = torch.from_numpy(np.stack(stroke_ids)).type(torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    lmasks = torch.tensor(lmasks, dtype=torch.float)
    lmasks = lmasks.type(torch.ByteTensor)
    return {"input_ids": input_ids, "length": lengthes, "input_mask": input_masks, "pinyin_ids": pinyin_ids, "stroke_ids": stroke_ids, "labels": labels, "lmask": lmasks}
