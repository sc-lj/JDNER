"""
@Time   :   2021-01-12 16:04:23
@File   :   dataset.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import load_json


class CorrectorDataset(Dataset):
    def __init__(self, fp):
        self.data = load_json(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']


def get_corrector_loader(fp, tokenizer, **kwargs):
    def _collate_fn(data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) or encoded_text[idx + move].startswith(
                        '##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels

    dataset = CorrectorDataset(fp)
    loader = DataLoader(dataset, collate_fn=_collate_fn, **kwargs)
    return loader
