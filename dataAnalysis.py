# -*- encoding: utf-8 -*-
'''
@File    :   dataAnalysis.py
@Time    :   2022/03/07 20:10:59
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''
import json
from collections import defaultdict
from pypinyin import pinyin, Style
import re


def read_sample_file():
    with open("data/train_500.txt", 'r') as f:
        lines = f.readlines()

    entities = defaultdict(list)
    samples = []
    sample = []
    texts = []
    space_char = "[unused1]"
    text_entity_pair = []
    entity_txt = ''
    entity_name = ''
    single_pair = defaultdict(list)
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            if len(sample):
                samples.append(sample)
                text = "".join([t.split("\t")[0] for t in sample])
                text = text.replace(space_char, " ")
                texts.append(text)
                if entity_name:
                    entities[entity_name].append("".join(entity_txt))
                    single_pair[entity_name].append("".join(entity_txt))
                text_entity_pair.append(
                    {"text": text, "entities": single_pair, "sample": sample})

            entity_txt = ''
            entity_name = ""
            single_pair = defaultdict(list)
            sample = []
            continue
        line = line.split(" ")

        if len(line) == 2:
            txt, tag = line
            sample.append("\t".join(line))
        elif len(line) == 1:
            tag = line[0]
            txt = " "
            sample.append("\t".join((space_char, line[0])))
        else:
            print(line)
        if tag == "O":
            if entity_name:
                entities[entity_name].append("".join(entity_txt))
                single_pair[entity_name].append("".join(entity_txt))
            entity_txt = ''
            entity_name = ""
        elif tag.startswith("B"):
            if entity_name:
                entities[entity_name].append("".join(entity_txt))
                single_pair[entity_name].append("".join(entity_txt))
            entity_txt = ''
            entity_txt += txt
            entity_name = tag.split("-")[-1]
        else:
            entity_txt += txt
            # entity_name = tag.split("-")[-1]

    if len(sample):
        text = "".join([t.split("\t")[0] for t in sample])
        text = text.replace(space_char, " ")
        texts.append(text)
        samples.append(sample)
        if entity_name:
            entities[entity_name].append("".join(entity_txt))
            single_pair[entity_name].append("".join(entity_txt))
        text_entity_pair.append(
            {"text": text, "entities": single_pair, "sample": sample})

    number = len(text_entity_pair)
    train_number = int(number*0.9)
    train_text_entity_pair = text_entity_pair[:train_number]
    val_text_entity_pair = text_entity_pair[train_number:]

    train_text = texts[:train_number]
    val_text = texts[train_number:]

    with open("data/text_entity_paires.json", 'w') as f:
        json.dump(text_entity_pair, f, ensure_ascii=False)

    with open("data/train.json", 'w') as f:
        json.dump(train_text_entity_pair, f, ensure_ascii=False)

    with open("data/val.json", 'w') as f:
        json.dump(val_text_entity_pair, f, ensure_ascii=False)

    with open("data/train.txt", 'w') as f:
        for line in train_text:
            f.write(line+"\n")

    with open("data/train.txt", 'w') as f:
        for line in val_text:
            f.write(line+"\n")

    with open("data/entites.json", 'w') as f:
        json.dump(entities, f, ensure_ascii=False)

    labels = entities.keys()
    labels = sorted(labels)
    new_labels = {}
    new_labels["O"] = 0
    num = 1
    for label in labels:
        new_labels["B-"+label] = num
        num += 1
        new_labels["I-"+label] = num
        num += 1
    label2ids = new_labels

    with open("data/label2ids.json", 'w') as f:
        json.dump(label2ids, f, ensure_ascii=False)

    return samples, texts, entities, text_entity_pair


chinses_compile = re.compile("[\u4e00-\u9fa5]")


def get_pinyin_vocab():
    """获取训练集中所有字符的拼音，或者多音拼音
    """
    with open("data/sample_text.txt", 'r') as f:
        lines = f.readlines()

    pinyin_vocab = []
    for line in lines:
        for w in line:
            if chinses_compile.match(w):
                w_pin = pinyin(w, style=Style.TONE3, heteronym=True)
                pinyin_vocab.extend(w_pin)
    pinyin_vocab = sum(pinyin_vocab, [])
    pinyin_vocab = set(pinyin_vocab)
    with open("pinyin_data/py_vocab.txt", 'w') as f:
        for line in pinyin_vocab:
            f.write(line+"\n")


if __name__ == "__main__":
    read_sample_file()
    # get_pinyin_vocab()
