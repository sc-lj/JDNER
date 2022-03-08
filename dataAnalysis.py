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


def read_sample_file():
    with open("data/train_500.txt", 'r') as f:
        lines = f.readlines()

    entities = defaultdict(list)
    samples = []
    sample = []
    texts = []
    special_char = "<SPACE>"
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
                text = text.replace(special_char, " ")
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
            sample.append("\t".join((special_char, line[0])))
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
        text = text.replace(special_char, " ")
        texts.append(text)
        samples.append(sample)
        if entity_name:
            entities[entity_name].append("".join(entity_txt))
            single_pair[entity_name].append("".join(entity_txt))
        text_entity_pair.append(
            {"text": text, "entities": single_pair, "sample": sample})

    with open("data/text_entity_paires.json", 'w') as f:
        json.dump(text_entity_pair, f, ensure_ascii=False)

    with open("data/sample_text.json", 'w') as f:
        for line in texts:
            f.write(line+"\n")

    with open("data/entites.json", 'w') as f:
        json.dump(entities, f, ensure_ascii=False)

    labels = entities.keys()
    labels = sorted(labels)
    label2ids = {label: i for i, label in enumerate(labels)}

    with open("data/label2ids.json", 'w') as f:
        json.dump(label2ids, f, ensure_ascii=False)

    return samples, texts, entities, text_entity_pair


if __name__ == "__main__":
    read_sample_file()
