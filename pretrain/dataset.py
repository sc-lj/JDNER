"""
@Time   :   2021-01-12 16:04:23
@File   :   dataset.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import re
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import load_json, read_file
from pinyin_tool import PinyinTool
from transformers.models.bert.tokenization_bert import BertTokenizer
from mask import PinyinConfusionSet, StrokeConfusionSet, Mask


# 空格前后都不是字母的场景
no_english_space = re.compile("([^a-zA-Z])\s+([^a-zA-Z])")
# 空格前是字母，后面不是字母的场景
no_english_space1 = re.compile("([a-zA-Z])\s+([^a-zA-Z])")
# 空格前不是字母，后面是字母的场景
no_english_space2 = re.compile("([^a-zA-Z])\s+([a-zA-Z])")
no_chinses_english_compile = re.compile("[^\u4e00-\u9fa5A-Za-z\s]")


def drop_space(text):
    """去掉非字母之间的所有空格

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    text = no_english_space.sub(r"\1\2", text)
    text = no_english_space1.sub(r"\1\2", text)
    text = no_english_space2.sub(r"\1\2", text)
    text = no_english_space.sub(r"\1\2", text)
    text = no_english_space.sub(r"\1\2", text).strip()
    return text


class NerDataset(Dataset):
    def __init__(self, fp, args):
        super().__init__()
        self.data = read_file(fp)
        self.space_char = "[unused1]"

        py_dict_path = './pinyin_data/zi_py.txt'
        py_vocab_path = './pinyin_data/py_vocab.txt'
        sk_dict_path = './stroke_data/zi_sk.txt'
        sk_vocab_path = './stroke_data/sk_vocab.txt'
        label2id_path = args.label_file
        with open(label2id_path, 'r') as f:
            self.label2ids = json.load(f)

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

        self.label_list = {}
        for key in self.tokenizer.vocab:
            self.label_list[self.tokenizer.vocab[key]] = key

        self.tokenid_pyid = {}
        self.tokenid_skid = {}
        for key in self.tokenizer.vocab:
            self.tokenid_pyid[self.tokenizer.vocab[key]
                              ] = self.pytool.get_pinyin_id(key)
            self.tokenid_skid[self.tokenizer.vocab[key]
                              ] = self.sktool.get_pinyin_id(key)
        self.max_sen_len = 512
        tokenid_pyid = {}
        tokenid_skid = {}
        for key in self.tokenizer.vocab:
            tokenid_pyid[self.tokenizer.vocab[key]
                         ] = self.pytool.get_pinyin_id(key)
            tokenid_skid[self.tokenizer.vocab[key]
                         ] = self.sktool.get_pinyin_id(key)

        same_py_file = './confusions/same_pinyin.txt'
        simi_py_file = './confusions/simi_pinyin.txt'
        stroke_file = './confusions/same_stroke.txt'
        tokenizer = self.tokenizer
        pinyin = PinyinConfusionSet(tokenizer, same_py_file)
        jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
        print('pinyin conf size:', len(pinyin.confusion))
        print('jinyin conf size:', len(jinyin.confusion))
        stroke = StrokeConfusionSet(tokenizer, stroke_file)
        self.masker = Mask(same_py_confusion=pinyin, simi_py_confusion=jinyin,
                           sk_confusion=stroke, tokenid2pyid=tokenid_pyid, tokenid2skid=tokenid_skid)

    def get_label_list(self):
        return self.label_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        tokens = drop_space(sample)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]

        _tokens = []
        # _labels = []
        _lmask = []
        pinyin_ids = []
        # stroke_ids = []
        _tokens.append("[CLS]")
        _lmask.append(0)
        # _labels.append(self.label2ids["O"])
        pinyin_ids.append(np.zeros(self.pylen))
        # stroke_ids.append(np.zeros(self.sklen))
        for token in tokens:
            if len(token.strip()) == 0:
                continue
            _tokens.append(token.lower())
            # _labels.append(self.label2ids[label])
            _lmask.append(1)
            pyid = self.pytool.get_pinyin_id(token)
            pinyin_ids.append(self.PYID2SEQ[pyid, :])
            # skid = self.sktool.get_pinyin_id(token)
            # stroke_ids.append(self.SKID2SEQ[skid, :])
        _tokens.append("[SEP]")
        # _labels.append(self.label2ids["O"])
        _lmask.append(0)
        pinyin_ids.append(np.zeros(self.pylen))
        # stroke_ids.append(np.zeros(self.sklen))
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
            # stroke_ids.append(np.zeros(self.sklen))
            # _labels.append(self.label2ids["O"])
            _lmask.append(0)
        pinyin_ids = np.vstack(pinyin_ids)
        # stroke_ids = np.vstack(stroke_ids)
        masked_ids, masked_flgs, masked_py_ids, masked_sk_ids = self.masker.mask_process(
            input_ids)
        lmask = masked_flgs*np.array(_lmask)
        label_ids = input_ids
        input_ids = masked_ids
        masked_pinyin_ids = masked_py_ids
        return {"input_ids": input_ids, "length": length, "input_mask": input_mask, "pinyin_ids": pinyin_ids,
                "masked_pinyin_ids": masked_pinyin_ids, "masked_sk_ids": masked_sk_ids, "labels": label_ids, "lmask": lmask}

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
    masked_pinyin_ids = []
    masked_sk_ids = []
    labels = []
    lmasks = []
    lengthes = []
    for batch in batches:
        input_ids.append(batch['input_ids'])
        input_masks.append(batch['input_mask'])
        pinyin_ids.append(batch['pinyin_ids'])
        masked_pinyin_ids.append(batch['masked_pinyin_ids'])
        masked_sk_ids.append(batch['masked_sk_ids'])
        labels.append(batch['labels'])
        lmasks.append(batch['lmask'])
        lengthes.append(batch['length'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.float32)
    pinyin_ids = torch.from_numpy(np.stack(pinyin_ids)).type(torch.long)
    masked_pinyin_ids = torch.from_numpy(
        np.stack(masked_pinyin_ids)).type(torch.float32)
    masked_sk_ids = torch.from_numpy(
        np.stack(masked_sk_ids)).type(torch.float32)
    labels = torch.tensor(labels, dtype=torch.float)
    lmasks = torch.tensor(lmasks, dtype=torch.float)
    # lmasks = lmasks.type(torch.ByteTensor)
    return {"input_ids": input_ids, "length": lengthes, "input_mask": input_masks, "pinyin_ids": pinyin_ids, "masked_pinyin_ids": masked_pinyin_ids,
            "masked_sk_ids": masked_sk_ids, "labels": labels, "lmask": lmasks}
