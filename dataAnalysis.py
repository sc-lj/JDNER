# -*- encoding: utf-8 -*-
'''
@File    :   dataAnalysis.py
@Time    :   2022/03/07 20:10:59
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

from flashtext import KeywordProcessor
import json
from collections import defaultdict
from pypinyin import pinyin, Style, lazy_pinyin
from simhash import Simhash, SimhashIndex
import re
import os
from random import shuffle
from tqdm import tqdm
# from LAC import LAC
from collections import defaultdict


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
    index = 0
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
                    {"id": index, "text": text, "entities": single_pair, "sample": sample})
                index += 1

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
            {"id": index, "text": text, "entities": single_pair, "sample": sample})

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
    # path = "/mnt/disk2/trainData/textData/文本分类数据/wiki_zh"
    path = "/mnt/disk2/data/wiki_zh_2019/wiki_zh"
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
    # all_data = wiki_data+news2016
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


def collect_duplicates_entity_data():
    """收集一个文本出现在多个实体类别中
    """

    duplicates_entities = set()
    with open("data/entites.json", 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        for k1, v1 in data.items():
            if k == k1:
                continue
            inter = set(v).intersection(set(v1))
            if len(inter) > 0:
                duplicates_entities.update(inter)
                # print(k, k1, inter)

    with open("data/train.json", 'r') as f:
        train_lines = json.load(f)
    with open("data/val.json", 'r') as f:
        val_lines = json.load(f)
    lines = train_lines+val_lines
    duplicates_result = defaultdict(
        lambda: defaultdict(lambda: {"text": [], "num": 0}))
    for line in lines:
        text = line['text']
        entities = line['entities']
        for lab, ent in entities.items():
            for en in set(ent):
                if en in duplicates_entities:
                    duplicates_result[en][lab]["num"] += 1
                    duplicates_result[en][lab]["text"].append(text)
    # duplicates_result = {en: {lab: list(text)for lab, text in lab_ent.items()}
    #                      for en, lab_ent in duplicates_result.items()}
    with open("data/duplicate_entity_text.json", 'w') as f:
        json.dump(duplicates_result, f, ensure_ascii=False)


def get_mul_label_info(train_lines, keywords_entities):
    """获取每个标签的标注的信息

    Args:
        train_lines (_type_): _description_
        keywords_entities (_type_): _description_
    """
    space_char = "[unused1]"
    correct_result = defaultdict(dict)
    for line in tqdm(train_lines):
        text = line['text']
        samples = line['sample']
        correct_result[text]['root'] = samples
        for lab, keywords in keywords_entities.items():
            extract_words = keywords.extract_keywords(text, span_info=True)
            label_info = []
            last_start = 0
            # tokens = [space_char if len(
            #     t.strip()) == 0 else t.lower() for t in text]
            for word, start, end in extract_words:
                label_info.append((lab, start, end))
            #     if last_start < start:
            #         O_token = tokens[last_start:start]

            #         label_info.extend(["\t".join((t, "O"))
            #                           for t in O_token])
            #     entity_token = tokens[start:end]
            #     label_info.extend(["\t".join((t, "B-"+lab)) if i == 0 else "\t".join((t, "I-"+lab))
            #                        for i, t in enumerate(entity_token)])
            #     last_start = end
            # O_token = tokens[last_start:len(tokens)]
            # label_info.extend(["\t".join((t, "O")) for t in O_token])
            correct_result[text][lab] = label_info
    return correct_result


def drop_label_info(root_label, other_label):
    """过滤一些与人工标注的实体标签有些不一致的字典标注的标签

    Args:
        root_label (_type_): _description_
        other_label (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_other_label = []
    for lab, start, end in other_label:
        root_lab = root_label[start:end]
        # 字典匹配的标签，第一个字符要与原始的人工标注的标记的边界一致，这是因为假设人工标注至少不会错误分词
        if not(root_lab[0].startswith("B") or root_lab[0].startswith("O")):
            continue
        if end < len(root_label) and root_label[end].startswith("I"):
            continue
        # 如果字典匹配的实体标签不在人工标注的实体标签之内，过滤
        labs = [lab[2:] for lab in root_lab if lab != "O"]
        if len(labs) and (lab not in labs):
            continue
        new_other_label.append([lab, start, end])

    if len(new_other_label) == 0:
        return new_other_label
    # 如果start相同，过滤短的
    other_label = [new_other_label[0]]
    for lab, start, end in new_other_label[1:]:
        if other_label[-1][1] == start and other_label[-1][2] < end:
            other_label[-1] = [lab, start, end]
            continue
        if other_label[-1][1] < start and other_label[-1][2] == end:
            continue
        other_label.append([lab, start, end])
    return other_label


def overlap_label_info(other_label, root_label):
    """判断字典匹配的重叠场景,暂时不考虑字典匹配重叠的场景，
    """
    overlap_label = []
    overlap = [other_label[0]]
    for lab, start, end in other_label[1:]:
        if overlap[-1][1] <= start < overlap[-1][2]:
            overlap.append([lab, start, end])
            continue
        if len(overlap) > 1:
            overlap_label.append(overlap)
        overlap = [[lab, start, end]]
    if len(overlap) > 1:
        overlap_label.append(overlap)

    overlap_label = sum(overlap_label, [])
    no_overlap_ = []
    for lab in other_label:
        if lab not in overlap_label:
            no_overlap_.append(lab)
    return no_overlap_


def merge_mul_label_info(correct_result):
    """合并每个标签标注的信息，以原始的人工标注为基准，

    Args:
        correct_result (_type_): _description_
    """
    corrected_data = []
    for text, label_info in correct_result.items():
        root = label_info.pop("root")
        root_label = [l.strip().split("\t")[-1] for l in root]
        root_token = [l.strip().split("\t")[0] for l in root]
        other_label = []
        for lab, info in label_info.items():
            other_label.extend(info)

        other_label = sorted(other_label, key=lambda x: (x[1], x[2]))
        other_label = drop_label_info(root_label, other_label)
        if len(other_label) == 0:
            sample = ["\t".join((t, l))
                      for t, l in zip(*(root_token, root_label))]
            corrected_data.append({"text": text, "sample": sample})
            continue
        other_label = overlap_label_info(other_label, root_label)
        """
        示例1，['网', '红', '直', '播', '支', '架']=>['B-14', 'I-14', 'B-5', 'I-5', 'B-4', 'I-4']
        字典匹配的：['网', '红', '直', '播']=>['B-5', 'I-5', 'I-5', 'I-5'],
                  ['直', '播', '支', '架']=>['B-4', 'I-4', 'I-4', 'I-4'],
                  这种场景，保持与原有人工标注一致，不改变
        
        场景1，如果人工标注的为O,但是字典匹配的多个，以最长的字典匹配的标签为准
        场景2，如果人工标注为多个类别，但是而字典的最长匹配只标注了一个类别，且与人工标注的类别有重叠，且与不与其他字典匹配进行重叠(如示例1)，以字典为准，
        场景3，
        """
        root_span_ = build_entity(root_label)
        root_entity_num = defaultdict(int)
        for ent, _, _ in root_span_:
            root_entity_num[ent] += 1

        for lab, start, end in other_label:
            gold_label = [l[2:] if l !=
                          "O" else l for l in root_label[start:end]]
            if lab in gold_label and "O" in gold_label and (gold_label[0] == "O" or gold_label[-1] == "O"):
                root_label[start:end] = ["B-"+lab if i ==
                                         0 else "I-"+lab for i in range(end-start)]
                # print(text, root_token[start:end], root_label[start:end], lab)
            # elif len(set(gold_label)) >= 2:
            #     print(text, root_token[start:end], root_label[start:end], lab)
        correct_span_ = build_entity(root_label)
        correct_entity_num = defaultdict(int)
        for ent, _, _ in correct_span_:
            correct_entity_num[ent] += 1

        for lab, value in root_entity_num.items():
            if value != correct_entity_num.get(lab, 0):
                print(lab, value, correct_entity_num.get(lab, 0))
                print()
        sample = ["\t".join((t, l))
                  for t, l in zip(*(root_token, root_label))]
        corrected_data.append({"text": text, "sample": sample})
    return corrected_data


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
                tmp_ent = ""
            start = index
            tmp_ent = tag[2:]
        elif tag.startswith("O"):
            if tmp_ent:
                entities.append((tmp_ent, start, index))
                tmp_ent = ""
            start = index
    if tmp_ent:
        entities.append((tmp_ent, start, index))
        tmp_ent = ""
    return entities


def correct_text():
    """通过同一个实体类别中，所有实体，去所有文本中匹配，去最长的。
      比如，类别38： 有这些实体词["8plus","iphone8","7plus","iPhone 7p/8p","7p/8p","7p"]，去文中匹配，将某些短的匹配文本，修正为最长的匹配方式

    """
    with open("data/strong_entites.json", 'r') as f:
        data = json.load(f)

    keywords_entities = {}
    for lab, entities in data.items():
        keywords = KeywordProcessor()
        keywords.non_word_boundaries = set()
        keywords.add_keywords_from_list(list(entities))
        keywords_entities[lab] = keywords
    with open("data/train.json", 'r') as f:
        train_lines = json.load(f)

    with open("data/val.json", 'r') as f:
        val_lines = json.load(f)

    correct_result = get_mul_label_info(train_lines, keywords_entities)
    # with open("data/correct_result_tmp.json", "w") as f:
    #     json.dump(correct_result, f, ensure_ascii=False)
    # with open("data/correct_result_tmp.json", "r") as f:
    #     correct_result = json.load(f)

    corrected_data = merge_mul_label_info(correct_result)
    with open("data/train_corrected.json", 'w') as f:
        json.dump(corrected_data, f, ensure_ascii=False)

    correct_result = get_mul_label_info(val_lines, keywords_entities)
    corrected_data = merge_mul_label_info(correct_result)
    with open("data/val_corrected.json", 'w') as f:
        json.dump(corrected_data, f, ensure_ascii=False)


def choice_high_entity():
    """选择具有强标注的实体，即只要出现就被标注的比例很高，且不与其他实体类型冲突
    """
    with open("data/entites.json", 'r') as f:
        data = json.load(f)

    def del_conflict_entity(lab1, lab2, entities):
        """删除冲突实体
        """
        for ent in entities:
            data[lab1].remove(ent)
            data[lab2].remove(ent)

    label_name = list(data.keys())
    for i in range(len(label_name)):
        v1 = data[label_name[i]]
        for j in range(i+1, len(label_name)):
            v2 = data[label_name[j]]
            inter = set(v1).intersection(set(v2))
            del_conflict_entity(label_name[i], label_name[j], inter)

    keywords_entities = {}
    for lab, entities in data.items():
        keywords = KeywordProcessor()
        keywords.non_word_boundaries = set()
        entities = [l for l in entities if len(l) > 2]
        keywords.add_keywords_from_list(entities)
        keywords_entities[lab] = keywords

    with open("data/train.json", 'r') as f:
        train_lines = json.load(f)
    with open("data/val.json", 'r') as f:
        val_lines = json.load(f)

    lines = train_lines+val_lines

    for lab in tqdm(label_name):
        lab_keywords = keywords_entities[lab]
        lab_keywords_match_info = defaultdict(lambda: defaultdict(int))
        for line in lines:
            text = line['text']
            sample = line["sample"]
            extract_words = lab_keywords.extract_keywords(text, span_info=True)
            for w, start, end in extract_words:
                samp = sample[start:end]
                # 统计字典标注的，有多少比例是人工没有标注的
                samp_tag = [l.split("\t")[-1] for l in samp]
                samp_tag = [l if l == "O" else l[2:] for l in samp_tag]
                # O_tag = int(any([t == lab for t in samp_tag]))
                O_tag = sum([t == lab for t in samp_tag])
                lab_keywords_match_info[w]["num"] += len(w)
                lab_keywords_match_info[w]["O_num"] += O_tag
        strong_entities = []
        for w, info in lab_keywords_match_info.items():
            ratio = info['O_num']/info['num']
            if ratio > 0.88:
                strong_entities.append(w)
        # for ent in low_entities:
        #     data[lab].remove(ent)
        data[lab] = strong_entities
    with open("data/strong_entites.json", 'w') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    # read_sample_file()
    # get_pinyin_vocab()
    # pretrain_data()
    # baidu_lac()
    # collect_duplicates_entity_data()
    correct_text()
    # choice_high_entity()
