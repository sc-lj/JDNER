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
from pypinyin import pinyin, Style, lazy_pinyin
from simhash import Simhash, SimhashIndex
import re
import os
from random import shuffle
from tqdm import tqdm
# from LAC import LAC


def read_sample_file():
    with open("data/2022京东电商数据比赛/京东商品标题实体识别数据集/train_data/train.txt", 'r') as f:
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

    with open("data/val.txt", 'w') as f:
        for line in val_text:
            f.write(line+"\n")

    with open("data/entites.json", 'w') as f:
        entities = {k: list(set(v)) for k, v in entities.items()}
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

    entity2ids = {k: i for i, k in enumerate(labels)}
    with open("data/entity2ids.json", 'w') as f:
        json.dump(entity2ids, f, ensure_ascii=False)

    return samples, texts, entities, text_entity_pair


chinses_compile = re.compile("[\u4e00-\u9fa5]")


def get_pinyin_vocab():
    """获取训练集中所有字符的拼音，或者多音拼音
    """
    # with open("data/sample_text.txt", 'r') as f:
    #     lines = f.readlines()
    with open("/mnt/disk2/PythonProgram/NLPCode/PretrainModel/chinese_bert_base/vocab.txt", 'r') as f:
        lines = f.readlines()
    pinyin_vocab = []
    zi_py = []
    for line in lines:
        for w in line:
            if chinses_compile.match(w):
                w_pin = lazy_pinyin(w, style=Style.TONE3)
                pinyin_vocab.append(w_pin)
                zi_py.append((w, w_pin[0]))
    pinyin_vocab = sum(pinyin_vocab, [])
    pinyin_vocab = set(pinyin_vocab)
    with open("pinyin_data/py_vocab_1.txt", 'w') as f:
        for line in pinyin_vocab:
            f.write(line+"\n")
    with open("pinyin_data/zi_py_1.txt", 'w') as f:
        for line in zi_py:
            f.write("\t".join(line)+"\n")


sent_split = re.compile("[？？。！!]")


def cut_text(texts):
    """
    将文本切分成单个句子
    """
    new_sents = []
    for sent in texts.strip().split("\n"):
        sents = sent_split.split(sent)
        punct = sent_split.findall(sent)
        punct = punct+[""]

        for s, p in zip(*(sents, punct)):
            if len(s+p) < 5:
                continue
            new_sents.append(s+p)
    return new_sents


def preprocess_wiki_data():
    """处理wiki数据集
    """
    path = "/mnt/disk2/trainData/textData/文本分类数据/wiki_zh"
    all_sentences = []
    for root, dirs, filenames in os.walk(path):
        for files in filenames:
            filename = os.path.join(root, files)
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                text = line['text']
                sentences = cut_text(text)
                all_sentences.extend(sentences)
    return all_sentences


def preprocess_news2016():
    """new2016zh-新闻语料
    """
    path = "/mnt/disk2/trainData/textData/文本分类数据/new2016zh-新闻语料json版/news2016zh_train.json"
    all_sentences = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            line = json.loads(line)
            content = line['content']
            sentences = cut_text(content)
            all_sentences.extend(sentences)
            line = f.readline()
    return all_sentences


def pretrain_data():
    """为预训练模型准备数据集
    """
    wiki_data = preprocess_wiki_data()
    # news2016 = preprocess_news2016()
    all_data = wiki_data
    shuffle(all_data)
    # no_duplicates = [all_data[0]]
    # indexs = 0
    # first_sent = all_data[0]
    # # 建立索引
    # objs = [(str(indexs), Simhash(first_sent))]
    # index = SimhashIndex(objs, k=1)  # k是容忍度；k越大，检索出的相似文本就越多
    # # 检索
    # for sent in tqdm(all_data[1:]):
    #     s1 = Simhash(sent)
    #     if len(index.get_near_dups(s1)):
    #         continue
    #     # 增加新索引
    #     indexs += 1
    #     index.add(str(index), s1)
    #     no_duplicates.append(sent)
    number = len(all_data)
    val_number = int(number*0.01)
    train = all_data[val_number:]
    val = all_data[:val_number]
    with open("data/pretrain_train_data.txt", "w") as f:
        for line in tqdm(train):
            f.write(line+"\n")
    with open("data/pretrain_val_data.txt", "w") as f:
        for line in tqdm(val):
            f.write(line+"\n")
    print("所有数据:", number)
    print("验证数据:", val_number)
    # print("去重后数据:", len(no_duplicates))


def baidu_lac():
    """n	普通名词	f	方位名词	s	处所名词	nw	作品名
        nz	其他专名	v	普通动词	vd	动副词	vn	名动词
        a	形容词	ad	副形词	an	名形词	d	副词
        m	数量词	q	量词	r	代词	p	介词
        c	连词	u	助词	xc	其他虚词	w	标点符号
        PER	人名	LOC	地名	ORG	机构名	TIME	时间

    Args:
        text (_type_): _description_
    """
    lac = LAC()
    with open("data/pretrain_train_data.txt", 'r') as f:
        lines = f.readlines()

    number = len(lines)
    batch_size = 20
    all_text = []
    for i in tqdm(range(0, number, batch_size)):
        batch_text = lines[i:i+batch_size]
        batch_text = [t.strip() for t in batch_text]
        result = lac.run(batch_text)
        for text, label in result:
            text_label = []
            for w, l in zip(*(text, label)):
                text_label.append("\t".join((w, l)))
            all_text.append(text_label)
    with open("data/pretrain_train_data_label.txt", 'w') as f:
        for line in all_text:
            for l in line:
                f.write(l+"\n")
            f.write("\n")


if __name__ == "__main__":
    read_sample_file()
    # get_pinyin_vocab()
    # pretrain_data()
    # baidu_lac()
