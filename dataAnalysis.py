# -*- encoding: utf-8 -*-
'''
@File    :   dataAnalysis.py
@Time    :   2022/03/07 20:10:59
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''
from collections import defaultdict

def read_sample_file():
    with open("data/train_500.txt",'r') as f:
        lines = f.readlines()

    entities = defaultdict(list)
    samples = []
    sample = []
    texts = []
    special_char = "<SPACE>"
    entity_txt = ''
    entity_name = ''
    for i,line in enumerate(lines):
        line = line.strip()
        if len(line)==0:
            if len(sample):
                samples.append(sample)
                text = "".join([t.split("\t")[0] for t in sample])
                text  = text.replace(special_char," ")
                texts.append(text)
                if entity_name:
                    entities[entity_name].append("".join(entity_txt))
                entity_txt = ''
                entity_name=""
            sample = []
            continue
        line = line.split(" ")
        
        if len(line)==2:
            txt,tag = line
            sample.append("\t".join(line))
        elif len(line) ==1:
            tag = line[0]
            txt = " "
            sample.append("\t".join((special_char,line[0])))
        else:
            print(line)
        if tag=="O":
            if entity_name:
                entities[entity_name].append("".join(entity_txt))
            entity_txt = ''
            entity_name = ""
        elif tag.startswith("B"):
            if entity_name:
                entities[entity_name].append("".join(entity_txt))
            entity_txt = ''
            entity_txt+=txt
            entity_name = tag.split("-")[-1]
        else:
            entity_txt+=txt
            # entity_name = tag.split("-")[-1]


    if len(sample):
        text = "".join([t.split("\t")[0] for t in sample])
        text  = text.replace(special_char," ")
        texts.append(text)
        samples.append(sample)
        if entity_name:
            entities[entity_name].append("".join(entity_txt))
    return samples,texts,entities



