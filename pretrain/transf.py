# -*- encoding: utf-8 -*-
'''
@File    :   transf.py
@Time    :   2022/03/28 19:33:22
@Author  :   lujun
@Version :   1.0
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''
import torch
from main import parse_args
from models import JDNerTrainingModel

args = parse_args()

args.number_tag = 81
args.pylen, args.sklen = 6, 10
args.num_labels = 21128
args.py_num_labels = 430  # 实际语音词汇表里面只有417，外加PAD和UNK，即只有419，但是可以多设几个。


models = JDNerTrainingModel.load_from_checkpoint(
    "lightning_logs/version_1/checkpoints/epoch=4-step=20834.ckpt", arguments=args)

bert = models.bert
bert_static = bert.state_dict()
torch.save(bert_static, "lightning_logs/version_1/checkpoints/bert.pt")
# cls = models.cls
# cls_static = cls.state_dict()
# torch.save(cls_static, "lightning_logs/version_1/checkpoints/cls.pt")

# py_cls = models.py_cls
# py_cls_static = py_cls.state_dict()
# torch.save(py_cls_static, "lightning_logs/version_1/checkpoints/py_cls.pt")
