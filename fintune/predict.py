# -*- encoding: utf-8 -*-
'''
File    :   predict.py
Time    :   2022/03/24 21:32:10
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   None
'''

from flashtext import KeywordProcessor
from modelsCRF import JDNerModel
from GlobalPointerModel import GlobalPointerNerModel
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pinyin_tool import PinyinTool
import numpy as np
import argparse
import os
from tqdm import tqdm
from collections import defaultdict


class NerDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        with open(args.val_file, 'r') as f:
            lines = f.readlines()
        self.data = lines
        self.model_type = args.model_type
        self.space_char = "[unused1]"

        py_dict_path = './pinyin_data/zi_py.txt'
        py_vocab_path = './pinyin_data/py_vocab.txt'
        sk_dict_path = './stroke_data/zi_sk.txt'
        sk_vocab_path = './stroke_data/sk_vocab.txt'

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
        sample = self.data[index].strip("\n")
        tokens = [self.space_char if len(
            t.strip()) == 0 else t for t in sample]
        root_token = tokens
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]
        root_token = ["[CLS]"] + tokens + ["[SEP]"]

        _tokens = []
        _lmask = []
        pinyin_ids = []
        stroke_ids = []
        _tokens.append("[CLS]")
        _lmask.append(1)
        pinyin_ids.append(np.zeros(self.pylen))
        stroke_ids.append(np.zeros(self.sklen))
        for token in tokens:
            _tokens.append(token.lower())
            _lmask.append(1)
            pyid = self.pytool.get_pinyin_id(token)
            pinyin_ids.append(self.PYID2SEQ[pyid, :])
            skid = self.sktool.get_pinyin_id(token)
            stroke_ids.append(self.SKID2SEQ[skid, :])
        _tokens.append("[SEP]")
        _lmask.append(1)
        pinyin_ids.append(np.zeros(self.pylen))
        stroke_ids.append(np.zeros(self.sklen))

        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)

        length = len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * length

        return {"input_ids": input_ids, "length": length, "input_mask": input_mask, "pinyin_ids": pinyin_ids, "stroke_ids": stroke_ids,
                "lmask": _lmask, "pylen": self.pylen, "sklen": self.sklen, "tokens": root_token}


def collate_fn(batches):
    max_length = max([batch['length'] for batch in batches])
    input_ids = []
    input_masks = []
    pinyin_ids = []
    stroke_ids = []
    lmasks = []
    lengthes = []
    tokens = []
    for batch in batches:
        length = batch['length']
        pylen = batch["pylen"]
        sklen = batch['sklen']
        input_ids.append(batch['input_ids']+[0]*(max_length-length))
        input_masks.append(batch['input_mask']+[0]*(max_length-length))
        pinyin_id = batch['pinyin_ids'] + \
            [np.zeros(((max_length-length), pylen))]
        pinyin_id = np.vstack(pinyin_id)
        pinyin_ids.append(pinyin_id)
        stroke_id = batch['stroke_ids'] + \
            [np.zeros(((max_length-length), sklen))]
        stroke_id = np.vstack(stroke_id)
        stroke_ids.append(stroke_id)
        lmasks.append(batch['lmask']+[0]*(max_length-length))
        lengthes.append(batch['length'])
        tokens.append(batch['tokens'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.float32)
    pinyin_ids = torch.from_numpy(np.stack(pinyin_ids)).type(torch.long)
    stroke_ids = torch.from_numpy(np.stack(stroke_ids)).type(torch.long)
    lmasks = torch.tensor(lmasks, dtype=torch.float)
    lmasks = lmasks.type(torch.ByteTensor)

    return {"input_ids": input_ids, "length": lengthes, "input_mask": input_masks, "pinyin_ids": pinyin_ids, "stroke_ids": stroke_ids,
            "lmask": lmasks, "tokens": tokens}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cuda',
                        type=str, help="硬件，cpu or cuda")
    parser.add_argument("--gpu_index", default=0, type=int,
                        help='gpu索引, one of [0,1,2,3,...]')
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="是否加载训练保存的权重, one of [t,f]")
    parser.add_argument(
        '--bert_checkpoint', default='/mnt/disk2/PythonProgram/NLPCode/PretrainModel/chinese_bert_base', type=str)
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=100, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=200, type=int, help='批大小')
    parser.add_argument('--num_workers', default=16,
                        type=int, help='多少进程用于处理数据')
    parser.add_argument('--warmup_epochs', default=8,
                        type=int, help='warmup轮数, 需小于训练轮数')
    parser.add_argument('--is_bilstm', default=False,
                        type=bool, help='是否采用双向LSTM')
    parser.add_argument('--lstm_hidden', default=200,
                        type=int, help='定义LSTM的输出向量')
    parser.add_argument('--lstm', default=False,
                        type=bool, help='是否添加LSTM模块')
    parser.add_argument('--adv', default=None,
                        choices=[None, "fgm", "pgd"], help='对抗学习模块')
    parser.add_argument('--epsilon', default=0.5, type=float, help='对抗学习的噪声系数')
    parser.add_argument('--model_type', default="crf", choices=["crf", 'global'],
                        type=str, help='损失函数类型')
    parser.add_argument('--lr', default=1e-5, type=float, help='学习率')
    parser.add_argument('--no_bert_lr', default=1e-3,
                        type=float, help='非bert部分参数的学习率')
    parser.add_argument('--accumulate_grad_batches',
                        default=1,
                        type=int,
                        help='梯度累加的batch数')
    parser.add_argument('--mode', default='train', type=str,
                        help='代码运行模式，以此来控制训练测试或数据预处理，one of [train, test]')
    parser.add_argument('--loss_weight', default=0.8,
                        type=float, help='论文中的lambda，即correction loss的权重')
    parser.add_argument(
        "--label_file", default="data/label2ids.json", help="实体标签id", type=str)
    parser.add_argument(
        "--entity_label_file", default="data/entity2ids.json", help="实体标签id", type=str)
    parser.add_argument(
        "--train_file", default="data/train.json", help="训练数据集")
    parser.add_argument(
        "--val_file", default="data/val.json", help="验证集")
    parser.add_argument(
        "--entity_path", default="data/entites.json", help="实体数据集")
    parser.add_argument(
        "--loss_func", default="corrected_nll", help="采用的loss func场景")
    parser.add_argument(
        "--use_focal_loss", default=True, help="采用的loss func场景")
    arguments = parser.parse_args()
    return arguments


def build_entity_v1(tags):
    """构建实体span，不含O标签

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


def build_entity(tags):
    """构建实体span,包含O标签

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
            start = index
            tmp_ent = tag[2:]
        elif tag.startswith("O"):
            if tmp_ent and tmp_ent != "O":
                entities.append((tmp_ent, start, index))
                start = index
            tmp_ent = "O"
    if tmp_ent:
        entities.append((tmp_ent, start, index))
        tmp_ent = ""

    return entities


def predict_crf():
    args = parse_args()
    with open(args.label_file, 'r') as f:
        label2ids = json.load(f)
    id2label = {v: k for k, v in label2ids.items()}
    args.number_tag = len(label2ids)
    space_char = "[unused1]"
    # args.val_file = "data/2022京东电商数据比赛/京东商品标题实体识别数据集/preliminary_test_a/sample_per_line_preliminary_A.txt"
    args.val_file = "data/2022京东电商数据比赛/京东商品标题实体识别数据集/train_data/unlabeled_train_data.txt"
    # args.val_file = "data/train.txt"
    # args.val_file = "data/val.txt"

    val_data = NerDataset(args)
    valid_loader = DataLoader(val_data, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    args.pylen, args.sklen = val_data.pylen, val_data.sklen
    device = torch.device("cuda")
    model = JDNerModel(args)
    model_static = torch.load(
        "lightning_logs/version_5/checkpoints/crfner_model.pt")
    model.load_state_dict(model_static)
    model.eval()
    model.to(device)
    keywords_entities, all_keywords_entities = init_keywords_entiteis()
    token_tags = []
    for batch in tqdm(valid_loader):
        input_ids = batch['input_ids']
        token = batch['tokens']
        length = batch['length']
        max_sen_len = max(length)
        input_masks = batch['input_mask']
        pinyin_ids = batch['pinyin_ids']
        stroke_ids = batch['stroke_ids']
        lmasks = batch['lmask']
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        pinyin_ids = pinyin_ids.to(device)
        stroke_ids = stroke_ids.to(device)
        lmasks = lmasks.to(device)
        with torch.no_grad():
            predict, best_tags_score, _ = model.decode(input_ids, input_masks, lmask=lmasks,
                                                       max_sen_len=max_sen_len)
        for pred, score, tok, leng in zip(*(predict, best_tags_score, token, length)):
            p_tag = [id2label[l] for l in pred[:leng]]
            score = [round(s, 3) for s in score]
            tok_tag = list(zip(*(tok, p_tag, score)))
            tok_tag = [" ".join(map(str, t)) for t in tok_tag]
            tok_tag = tok_tag[1:-1]
            line = map_dict_label_O(tok_tag, keywords_entities)
            token_tags.append(tok_tag)

    # with open("data/2022京东电商数据比赛/京东商品标题实体识别数据集/preliminary_test_a/crf_predict_v1.txt", "w") as f:
    #     for line in token_tags:
    #         line = map_dict_label_O(line, keywords_entities)
    #         for l in line:
    #             l = l.replace(space_char, " ")
    #             f.write(l+"\n")
    #         f.write("\n")

    with open("data/2022京东电商数据比赛/京东商品标题实体识别数据集/train_data/unlabeled_train_data_predict.txt", "w") as f:
        for line in token_tags:
            line = map_dict_label_O(line, keywords_entities)
            # line = map_dict_label(line, keywords_entities,
            #                       all_keywords_entities)
            f.write("\n".join(line)+"\n")
            f.write(""+"\n")

    # with open("data/train_predict.txt", "w") as f:
    #     for line in token_tags:
    #         # line = map_dict_label_O(line, keywords_entities)
    #         f.write("\n".join(line)+"\n")
    #         f.write("\n")

    # with open("data/val_predict.txt", "w") as f:
    #     for line in token_tags:
    #         # line = map_dict_label_O(line, keywords_entities)
    #         f.write("\n".join(line)+"\n")
    #         f.write(""+"\n")


def init_keywords_entiteis():
    """初始化

    Returns:
        _type_: _description_
    """
    with open("data/strong_entites.json", 'r') as f:
        data = json.load(f)
    keywords = KeywordProcessor()
    keywords._white_space_chars = []
    keywords.non_word_boundaries = set()
    keywords.add_keywords_from_dict(data)
    with open("data/weak_strong_entites.json", 'r') as f:
        data = json.load(f)
    all_keywords = KeywordProcessor()
    all_keywords.non_word_boundaries = set()
    all_keywords._white_space_chars = []
    all_keywords.add_keywords_from_dict(data)
    return keywords, all_keywords


def get_mul_label_info(text, keywords_entities):
    """获取每个标签下实体字符硬匹配的标注的信息

    Args:
        line (_type_): _description_
        keywords_entities (_type_): _description_
    """
    space_char = "[unused1]"
    correct_result = []
    for lab, keywords in keywords_entities.items():
        extract_words = keywords.extract_keywords(text, span_info=True)
        label_info = []
        last_start = 0
        # tokens = [space_char if len(
        #     t.strip()) == 0 else t.lower() for t in text]
        for word, start, end in extract_words:
            label_info.append((lab, word, start, end))
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
        correct_result.extend(label_info)
    return correct_result


def overlap_label_info(O_label):
    """判断字典匹配的重叠场景,暂时不考虑字典匹配重叠的场景，
    """
    # 如果start相同，过滤短的
    other_label = []
    for lab, start, end in O_label:
        if len(other_label) == 0:
            other_label.append([lab, start, end])
            continue
        # 开始索引相同，结束索引不同，取结束索引最大的
        if other_label[-1][1] == start and other_label[-1][2] < end:
            other_label[-1] = [lab, start, end]
            continue
        # 开始索引不同，end索引相同，保留开始索引最小的
        if other_label[-1][1] < start and other_label[-1][2] == end:
            continue
        # 两个实体匹配刚好交叉，取第一个
        if other_label[-1][1] < start < other_label[-1][2] < end:
            continue
        # 如果有歧义，两者都不要
        if other_label[-1][1] == start and other_label[-1][2] == end:
            del other_label[-1]
            continue
        if other_label[-1][1] < start and end < other_label[-1][2]:
            continue
        other_label.append([lab,  start, end])

    return other_label


def map_dict_label(line, keywords_entities, all_keywords_entities):
    """用字典去匹配,如果与算法冲突，以字典为主

    Args:
        words (_type_): _description_
        keywords_entities (_type_): _description_
    """
    space_char = "[unused1]"
    text = [l.split(" ")[0] for l in line]
    text = "".join(text)
    text = text.replace(space_char, " ")
    tags = [l.split(" ")[1] for l in line]
    tag_index = build_entity(tags)
    # for lin in tag_index:
    #     lin.append("t")
    map_index = keywords_entities.extract_keywords(text, span_info=True)
    # for lin in map_index:
    #     lin.append("m")

    new_index = []
    for lin in map_index:
        l, s, e = lin
        temp = []
        for t_lin in tag_index:
            t_l, l_s, l_e = t_lin
            if l_e <= s:
                new_index.append(t_lin)
            elif l_s >= e:
                break
            else:
                temp.append(t_lin)
        if len(temp):
            new_temp = []
            label, start, end = l, s, e
            for t_lin in temp:
                t_l, l_s, l_e = t_lin
                if l_s < s < l_e:
                    if s-l_s >= 2:
                        info = all_keywords_entities.extract_keywords(
                            text[l_s:s], span_info=True)
                        if len(info) != 0:
                            lab, st, en = info[0]
                            st, en = l_s+st, l_s+en
                            new_temp.append([lab, st, en])

                elif s <= l_s < l_e <= e:
                    pass
                elif l_s < e < l_e:
                    if l_e-e >= 2:
                        info = all_keywords_entities.extract_keywords(
                            text[e:l_e], span_info=True)
                        if len(info) != 0:
                            lab, st, en = info[0]
                            st, en = e+st, e+en
                            new_temp.append([lab, st, en])
                else:
                    print()
            new_temp.append([label, start, end])
            new_temp = sorted(new_temp, key=lambda x: x[1])
            new_index.extend(new_temp)
    line = expand_text(new_index, text)
    return line


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
                        for i, t in enumerate(text[start:lab])])
        s = end

    if s < len(text):
        text_tag.extend([" ".join((t, "O")) for t in text[s:]])
    return text_tag


def map_dict_label_O(line, keywords_entities):
    """用字典去匹配连续为O的字符

    Args:
        words (_type_): _description_
        keywords_entities (_type_): _description_
    """
    space_char = "[unused1]"
    text = [l.split(" ")[0] for l in line]
    text = "".join(text)
    text = text.replace(space_char, " ")
    tags = [l.split(" ")[1] for l in line]
    tag_index = build_entity(tags)

    O_tag_text_index = [line for line in tag_index if line[0] == "O"]
    for O_label, start, end in O_tag_text_index:
        O_text = text[start:end]
        if len(O_text) < 2:
            continue
        map_result = keywords_entities.extract_keywords(O_text, span_info=True)
        if len(map_result) == 0:
            continue
        map_result = [l for l in map_result if (l[2]-l[1]) >= 2]

        for lab, start1, end1 in map_result:
            sub = end1-start1
            labs = ["B-"+lab if i == 0 else "I-"+lab for i in range(sub)]
            line[start+start1:start+end1] = [" ".join((lin.split(" ")[0], labs[i]))
                                             for i, lin in enumerate(line[start+start1:start+end1])]
    return line


def predict_global_pointer():
    args = parse_args()
    args.batch_size = 25
    with open(args.entity_label_file, 'r') as f:
        label2ids = json.load(f)
    id2label = {v: k for k, v in label2ids.items()}
    args.number_tag = len(label2ids)
    val_data = NerDataset(args)
    valid_loader = DataLoader(val_data, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, prefetch_factor=10, collate_fn=collate_fn)
    args.pylen, args.sklen = val_data.pylen, val_data.sklen
    device = torch.device("cuda")
    model = GlobalPointerNerModel(args)
    model_static = torch.load(
        "lightning_logs/version_2/checkpoints/global_pointer_model.pt")
    model.load_state_dict(model_static)
    model.eval()
    model.to(device)
    token_tags = []
    for batch in tqdm(valid_loader):
        input_ids = batch['input_ids']
        token = batch['tokens']
        length = batch['length']
        max_sen_len = max(length)
        input_masks = batch['input_mask']
        pinyin_ids = batch['pinyin_ids']
        stroke_ids = batch['stroke_ids']
        lmasks = batch['lmask']
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        pinyin_ids = pinyin_ids.to(device)
        stroke_ids = stroke_ids.to(device)
        lmasks = lmasks.to(device)
        with torch.no_grad():
            predict = model(input_ids, input_masks, max_sen_len=max_sen_len)
        predict = predict.cpu().numpy()
        pre_index = np.greater(predict, 0)
        batch_size = predict.shape[0]
        for j in range(batch_size):
            batch_token = token[j]
            pre_entity = []
            last_start = 0
            # 按start索引排序
            pre_index_sorted = sorted(
                zip(*(pre_index[j].nonzero())), key=lambda x: x[1])
            if len(pre_index_sorted):
                pre_index_sorted = merge_duplicate(pre_index_sorted)
                for lab, start, end in pre_index_sorted:
                    if last_start < start:
                        O_token = batch_token[last_start:start]
                        pre_entity.extend(["\t".join((t, "O"))
                                          for t in O_token])
                    entity_token = batch_token[start:end]
                    pre_entity.extend(["\t".join((t, "B-"+id2label[lab])) if i == 0 else "\t".join((t, "I-"+id2label[lab]))
                                       for i, t in enumerate(entity_token)])
                    last_start = end
            O_token = batch_token[last_start:len(batch_token)]
            pre_entity.extend(["\t".join((t, "O")) for t in O_token])
            pre_entity = pre_entity[1:-1]
            assert len(pre_entity) == len(batch_token)-2
            token_tags.append(pre_entity)
    with open("data/global_pointer_predict.txt", "w") as f:
        for line in token_tags:
            f.write("\t".join(line)+"\n")


def merge_duplicate(lines):
    """合并重叠的

    Args:
        lines (_type_): _description_
    """
    new_lines = [lines[0]]
    for lab, start, end in lines[1:]:
        if new_lines[-1][1] <= start < new_lines[-1][2]:
            # 表示重叠,重叠部分，不考虑标签的多样性，只去第一个标签
            new_lines[-1] = (new_lines[-1][0], min(start,
                             new_lines[-1][1]), max(new_lines[-1][2], end))
        else:
            new_lines.append((lab, start, end))
    return new_lines


def tranf_model_to_static():
    args = parse_args()
    args.pylen, args.sklen = 6, 10
    with open(args.label_file, 'r') as f:
        label2ids = json.load(f)

    with open(args.entity_label_file, 'r') as f:
        entity2ids = json.load(f)

    if args.model_type == "crf":
        ckpt_file = "lightning_logs/version_5/checkpoints/epoch=14-f1=0.7960-pre=0.785-recall=0.807.ckpt"
        ckpt_path = os.path.dirname(ckpt_file)
        from modelsCRF import CRFNerTrainingModel
        number_tag = len(label2ids)
        args.number_tag = number_tag
        # GlobalPointerNerTrainingModel.load_from_checkpoint()
        crfner_model = CRFNerTrainingModel.load_from_checkpoint(
            ckpt_file, arguments=args)
        crfner_model = crfner_model.model
        crfner_model_static = crfner_model.state_dict()
        torch.save(crfner_model_static, os.path.join(
            ckpt_path, "crfner_model.pt"))
    else:
        from GlobalPointerModel import GlobalPointerNerTrainingModel
        number_tag = len(entity2ids)
        args.number_tag = number_tag
        ckpt_file = "lightning_logs/version_2/checkpoints/epoch=43-f1=0.7711-pre=0.818-recall=0.729.ckpt"
        ckpt_path = os.path.dirname(ckpt_file)
        global_pointer_model = GlobalPointerNerTrainingModel.load_from_checkpoint(
            "lightning_logs/version_2/checkpoints/epoch=43-f1=0.7711-pre=0.818-recall=0.729.ckpt", arguments=args)
        global_pointer_model = global_pointer_model.model
        global_pointer_model_static = global_pointer_model.state_dict()
        torch.save(
            global_pointer_model_static, os.path.join(ckpt_path, "global_pointer_model.pt"))


if __name__ == "__main__":
    # tranf_model_to_static()
    predict_crf()
    # predict_global_pointer()
