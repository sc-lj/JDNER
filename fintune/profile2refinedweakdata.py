"""This file turns pofile into refined weakly labeled data."""
import numpy as np
import pickle
import os
import argparse
from scipy.stats import binned_statistic
from flashtext import KeywordProcessor
from tqdm import tqdm
import re
import json

parser = argparse.ArgumentParser(description='Profile to SelfTraining Data')
parser.add_argument('--weak_file', type=str,
                    default="weak", help='weak file name')
parser.add_argument('--wei_rule', type=str,
                    default="avgaccu_weak_non_O_promote", help='weighting rule')
parser.add_argument('--pred_rule', type=str,
                    default="non_O_overwrite", help='prediction rule')
args = parser.parse_args()


def statistic_predict_info():
    """利用crf对每个字符的分数，统计预测准确实体和错误实体的分数的分布情况
    """
    with open("data/train_predict.txt", 'r') as f:
        lines = f.readlines()

    with open("data/train.json", 'r') as f:
        target = json.load(f)

    def extract_line(line):
        scores = []
        texts = ""
        tags = []
        for l in line:
            word, tag, score = l.split(" ")
            texts += word
            scores.append(score)
            tags.append(tag)
        return texts, scores, tags

    predict_ = {}
    single_sample = []
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0 and len(single_sample):
            text, score, tags = extract_line(single_sample)
            predict_[text] = [score, tags]
            single_sample = []
        elif len(line) != 0:
            single_sample.append(line)
    if len(single_sample):
        text, score, tags = extract_line(single_sample)
        predict_[text] = [score, tags]

    number = 0
    dev_profile_data = []
    for line in target:
        sample = line['sample']
        text = "".join([t.split("\t")[0] for t in sample])
        target_tag = [t.split("\t")[1] for t in sample]
        predict_score, predict_tag = predict_[text]
        predict_score = list(map(float, predict_score))
        max_score = sum(predict_score)/len(predict_score)
        is_correct = False
        if target_tag == predict_tag:
            is_correct = True
            number += 1
        dev_profile_data.append([is_correct, max_score])

    return dev_profile_data


dev_profile_data = statistic_predict_info()


print(len(dev_profile_data))

# Averaged Score

nbins = 50
dev_profile_data.sort(key=lambda x: x[1])
scores = np.array([float(x[1]) for x in dev_profile_data])
acu = np.array([1.0 if x[0] else 0.0 for x in dev_profile_data])
print('averge query level accu: ', sum(acu) / len(acu))

bins = scores[::len(scores) // nbins]
bins[0] = -10000
bins[-1] = 10000

bin_means, bin_edges, binnumber = binned_statistic(
    scores, acu, statistic='mean', bins=bins)

cum_accu_lower = [sum(bin_means[:i + 1]) / (i + 1)
                  for i in range(len(bin_means))]
cum_accu_higher = [sum(bin_means[-i - 1:]) / (i + 1)
                   for i in range(len(bin_means) - 1, -1, -1)]
assert len(cum_accu_higher) == len(cum_accu_lower) == len(bin_means)


# Define mapping from exampel to weight base on wei_rule
if args.wei_rule == "avgaccu":
    def mapex2wei(ex):
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[0] >= edge:
                return wei
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == "avgaccu_weak_non_O_promote":
    def mapex2wei(ex):
        ps = ex[1]
        ls = ex[2]
        prop = 0.0
        for p, l in zip(ps, ls):
            if l != 'O':
                prop += 1
        prop /= len(ps)
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[0] >= edge:
                return wei * (1 - prop) + prop
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == "corrected":
    def mapex2wei(ex):
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[0] >= edge:
                return 2 * wei - 1
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == "corrected_weak_non_O_promote":
    def mapex2wei(ex):
        ps = ex[1]
        ls = ex[2]
        prop = 0.0
        for p, l in zip(ps, ls):
            if l != 'O':
                prop += 1
        prop /= len(ps)
        for edge, wei in zip(bin_edges[-2::-1], bin_means[::-1]):
            if ex[0] >= edge:
                return (2 * wei - 1) * (1 - prop) + prop
        raise RuntimeError("Not catched by rule")
elif args.wei_rule == 'uni':
    def mapex2wei(ex):
        return 1
elif re.match(r'wei_accu_pairs(-\d.\d_\d\d)*-\d.\d', args.wei_rule):
    wei_accu = args.wei_rule.split('-')[1:]
    wei_accu[-1] += '_100'
    wei_accu = [x.split('_') for x in wei_accu]
    wei_accu = [(float(w), float(a)) for w, a in wei_accu]

    def mapex2wei(ex):
        for wei, edge in wei_accu:
            if ex[0] <= edge:
                return wei
        raise RuntimeError("Not catched by rule")
else:
    raise NotImplementedError(f"{args.wei_rule} not implemented")


def extract_line(line):
    scores = []
    texts = ""
    tags = []
    space_char = "[unused1]"
    first_word = line[0].split(" ")[0].strip()
    while first_word == space_char:
        line = line[1:]
        first_word = line[0].split(" ")[0].strip()
    last_word = line[-1].split(" ")[0].strip()
    while last_word == space_char:
        line = line[:-1]
        last_word = line[-1].split(" ")[0].strip()

    for l in line:
        if len(l.split(" ")) == 2:
            word, tag = l.split(" ")
            score = 1
        else:
            word, tag, score = l.split(" ")
        texts += word
        scores.append(score)
        tags.append(tag)
    return texts, scores, tags


def expand_text(index, text):
    """_summary_

    Args:
        index (_type_): _description_
        text (_type_): _description_
    """
    s = 0
    text_tag = []
    for lab, start, end in index:
        if start > s:
            text_tag.extend([" ".join((t, "O")) for t in text[s:start]])
        text_tag.extend([" ".join((t, "B-"+lab)) if i == 0 else " ".join((t, "I-"+lab))
                        for i, t in enumerate(text[start:end])])
        s = end

    if s < len(text):
        text_tag.extend([" ".join((t, "O")) for t in text[s:]])
    return text_tag


"""字典匹配
"""

with open("data/strong_entites.json", 'r') as f:
    data = json.load(f)
keywords = KeywordProcessor()
keywords._white_space_chars = []
keywords.non_word_boundaries = set()
keywords.add_keywords_from_dict(data)

with open("data/2022京东电商数据比赛/京东商品标题实体识别数据集/train_data/unlabeled_train_data_predict.txt", 'r') as f:
    lines = f.readlines()


space_char = "[unused1]"

weak_profile_data = {}
single_sample = []
for line in tqdm(lines):
    line = line.strip()
    if len(line) == 0 and len(single_sample):
        text, score, tags = extract_line(single_sample)
        sample = ["\t".join((t, la)) for t, la in zip(*(text, tags))]
        drop_text = text.replace(space_char, " ")
        info = keywords.extract_keywords(drop_text, span_info=True)
        map_tags = expand_text(info, line)
        target_tag = [t.split(" ")[-1] for t in map_tags]
        score = map(float, score)
        max_score = max(score)
        weak_profile_data[drop_text] = [[max_score, tags, target_tag], sample]
        single_sample = []
    elif len(line) != 0:
        single_sample.append(line)

if len(single_sample):
    text, score, tags = extract_line(single_sample)
    sample = ["\t".join((t, la)) for t, la in zip(*(text, tags))]
    drop_text = text.replace(space_char, " ")
    info = keywords.extract_keywords(drop_text, span_info=True)
    map_tags = expand_text(info, line)
    target_tag = [t.split(" ")[-1] for t in map_tags]
    score = map(float, score)
    max_score = max(score)
    weak_profile_data[drop_text] = [[max_score, tags, target_tag], sample]

print(len(weak_profile_data))

print("==Generating Weights==")
weights = []
for txt, ex in weak_profile_data.items():
    ex, sample = ex
    weight = mapex2wei(ex)
    weights.append({"sample": sample, "weight": weight, "text": txt})

with open("data/2022京东电商数据比赛/京东商品标题实体识别数据集/train_data/unlabeled_train_data_weak.json", 'w') as f:
    json.dump(weights, f, ensure_ascii=False)


# Rule for generating refined labels.
def save_rule(rule, pred, label, score):
    if "-" in rule:
        rule = rule.split('-')[1]
    if rule is None or rule == 'no':
        return label
    elif rule == 'non_O_overwrite':
        if label != 'O':
            return label
        else:
            return pred
    elif re.match(r'non_O_overwrite_over_accu_\d\d', rule):
        if label != 'O':
            return label
        thre = int(rule[-2:]) / 100.
        for edge, accu in zip(bin_edges[-2::-1], cum_accu_lower[::-1]):
            if ex[1] >= edge:
                if accu < thre:
                    return 'X'
                else:
                    return pred
        assert False, "accu must be found"
    elif re.match(r'non_O_overwrite_all_overwrite_over_accu_\d\d', rule):
        if label == 'O':
            return pred
        thre = int(rule[-2:]) / 100.
        for edge, accu in zip(bin_edges[-2::-1], cum_accu_lower[::-1]):
            if ex[1] >= edge:
                if accu < thre:
                    return label
                else:
                    return pred
        assert False, "accu must be found"
    elif rule == 'all_overwrite':
        return pred
    else:
        raise NotImplementedError(rule + ' not implemented')


def screen_rule(rule, ps, ls, score):
    """Select sample or not."""
    ori_rule = rule
    if rule is None or rule == 'no' or '-' not in rule:
        return True
    else:
        rule = rule.split("-")[0]
        if rule == 'drop_allmatch':
            for p, l in zip(ps, ls):
                if l != 'O' and p != l:
                    return True
            return False
        if rule == 'drop_allmatch_error':
            prevp = None
            for p, l in zip(ps, ls):
                p = save_rule(ori_rule, p, l)
                if p.startswith('I-'):
                    if prevp != p and prevp != p.replace('I-', 'B-'):
                        return False
                prevp = p
            for p, l in zip(ps, ls):
                if l != 'O' and p != l:
                    return True
            return False
        else:
            raise NotImplementedError(rule + " not implemented")


# total_error_nums = 0
# total_nomatch_nums = 0
# total_save_nums = 0
# with open(TXTSAVEPATH, 'w') as fout:
#     print("==Generating Labels==")
#     for ex in tqdm(weak_profile_data):
#         score = ex[1]
#         ps = ex[-3]
#         ls = ex[-2]
#         es = ex[-1]
#         if not screen_rule(args.pred_rule, ps, ls, score):
#             continue
#         total_save_nums += 1
#         prevp = 'O'
#         preve = None
#         for p, l, e in zip(ps, ls, es):
#             if p != l and l != 'O':
#                 total_nomatch_nums += 1
#             p = save_rule(args.pred_rule, p, l, score)
#             if p.startswith('I-'):
#                 if prevp != p and prevp != p.replace('I-', 'B-'):
#                     total_error_nums += 1
#             prevp = p
#             preve = e
#             fout.write("{}\t{}\n".format(e, p))
#         fout.write("\n")
# print("Total # of Saves ", total_save_nums, " / ", len(weak_profile_data))
# print("Total # of Errors ", total_error_nums)
# print("Total # of Not Match Weak ", total_nomatch_nums)
