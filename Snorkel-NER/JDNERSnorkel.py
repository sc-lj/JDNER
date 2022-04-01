# -*- encoding: utf-8 -*-
'''
File    :   JDNERSnorkel.py
Time    :   2022/03/24 18:37:58
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   None
'''
from nltk.corpus import wordnet
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from sklearn.metrics import confusion_matrix
from snorkel.analysis import get_label_buckets
from snorkel.labeling import PandasLFApplier, LFApplier, LFAnalysis, labeling_function
from nltk import word_tokenize
import pandas as pd
import numpy as np
import re
import editdistance as ed
from sklearn.model_selection import train_test_split
import nltk
import spacy
nltk.download('wordnet')
nltk.download('punkt')
