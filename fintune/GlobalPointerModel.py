# -*- encoding: utf-8 -*-
'''
File    :   GlobalPointerModel.py
Time    :   2022/03/23 23:21:07
Author  :   lujun
Version :   1.0
Contact :   779365135@qq.com
License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
Desc    :   None
'''
import os
import json
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel as bertModel, BertPooler, BertOnlyMLMHead, BertEmbeddings as bertEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin
from torchcrf import CRF
from AdTraings import PGD, FGM


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(self, output_dim, merge_mode='mul', **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(
            0, seq_len, dtype=torch.float, device=inputs.device).reshape(1, -1)

        indices = torch.arange(0, self.output_dim // 2,
                               dtype=torch.float, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        # [1,seq_len,output_dim//2]
        embeddings = torch.matmul(
            position_ids.unsqueeze(-1), indices.unsqueeze(0))
        # [1,seq_len,output_dim//2,2]
        embeddings = torch.stack(
            [torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # [1,seq_len,output_dim]
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            embeddings = embeddings.repeat([batch_size, 1, 1])
            return torch.cat([inputs, embeddings], dim=-1)


class EmbeddingNetwork(nn.Module):
    def __init__(self, config, PYLEN, num_embeddings, max_sen_len=512):
        super().__init__()
        self.config = config
        self.PYDIM = 32
        self.seq_len = PYLEN
        num_embeddings = num_embeddings
        self.pyemb = nn.Embedding(num_embeddings, self.PYDIM)
        self.GRU = nn.GRU(
            self.PYDIM,
            self.config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=False,
        )
        self.MAX_SEN_LEN = max_sen_len

    def forward(self, sen_pyids, max_sen_len):
        sen_pyids = sen_pyids.reshape(-1, self.seq_len)
        sen_emb = self.pyemb(sen_pyids)
        sen_emb = sen_emb.reshape(-1, self.seq_len, self.PYDIM)
        all_out, final_out = self.GRU(sen_emb)
        final_out = final_out.mean(0, keepdim=True)
        lstm_output = final_out.reshape(
            shape=[-1, max_sen_len, self.config.hidden_size])

        return lstm_output


class BertEmbeddings(bertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.position_embeddings = SinusoidalPositionEmbedding(
            output_dim=config.hidden_size)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, pinyin_embs=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:,
                                             past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # if self.position_embedding_type == "absolute":
        #     position_embeddings = self.position_embeddings(position_ids)
        #     embeddings += position_embeddings
        embeddings = self.position_embeddings(embeddings)
        if pinyin_embs is not None:
            embeddings += pinyin_embs
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(bertModel):
    def __init__(self, config, args):
        super().__init__(config)
        PYLEN, SKLEN = args.pylen, args.sklen
        self.config = config
        self.py_emb = EmbeddingNetwork(
            self.config, PYLEN=PYLEN, num_embeddings=30)
        self.sk_emb = EmbeddingNetwork(
            self.config, PYLEN=SKLEN, num_embeddings=7)
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        py2ids=None,
        sk2ids=None,
        max_sen_len=None
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        device = input_ids.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)
        pinyin_emb = None
        if py2ids is not None:
            py_emb = self.py_emb(py2ids, max_sen_len)
            pinyin_emb = py_emb
        if sk2ids is not None:
            sk_emb = self.sk_emb(sk2ids, max_sen_len)
            if pinyin_emb is not None:
                pinyin_emb += sk_emb
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            pinyin_embs=pinyin_emb)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        return sequence_output


# 句子mask
def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        # return x * mask + value * (1 - mask)
        return mask


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    mask = (mask.unsqueeze(1).unsqueeze(-1)*mask.unsqueeze(1).unsqueeze(1))
    # logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    # logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # logits = torch.where(torch.isnan(
    #     logits), torch.full_like(logits, -1e4), logits)
    value = -1e12
    logits = logits * mask + value * (1 - mask)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, self.head_size * self.heads * 2)

    def forward(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_size * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
            sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        #  qw.permute(0, 2, 1, 3) [batch_size,heads,seq_len,head_size]
        # 计算内积
        # [batch_size,heads,seq_len,seq_len]
        logit = torch.matmul(qw.permute(0, 2, 1, 3), kw.permute(0, 2, 3, 1))
        # 计算内积
        # logit = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logit, mask)
        # scale返回
        return logits / self.head_size ** 0.5


# 搭建网络，就是输入bert获取句子向量特征表示，在用全局指针来获取一个矩阵标注，[batch_size ,class_count ,sentence_length ,sentence_length]
class JDNerModel(nn.Module):
    def __init__(self, args):
        super(JDNerModel, self).__init__()
        self.args = args
        num_labels = self.args.number_tag
        self.config = BertConfig.from_pretrained(self.args.bert_checkpoint)
        self.bert = BertModel(self.config, args)
        self.head = GlobalPointer(num_labels, 8, self.config.hidden_size)
        self.lstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size //
                            2, num_layers=1,  bidirectional=True, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, max_sen_len=None):
        x1 = self.bert(
            input_ids=input_ids, attention_mask=input_mask, py2ids=pinyin_ids, sk2ids=stroke_ids, max_sen_len=max_sen_len)

        x2, (_, _) = self.lstm(x1)
        x2 = self.dropout(x2)
        logits = self.head(x2, mask=input_mask)
        return logits

    def load_from_transformers_state_dict(self, gen_fp):
        """
            从transformers加载预训练权重
            :param gen_fp:
            :return:
        """
        state_dict = OrderedDict()
        gen_state_dict = torch.load(gen_fp)
        for k, v in gen_state_dict.items():
            name = k
            # if name.startswith('bert'):
            #     name = name[5:]
            # if name.startswith('encoder'):
            #     name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)


class GlobalPointerNerTrainingModel(pl.LightningModule):
    def __init__(self, arguments):
        super().__init__()
        self.args = arguments
        self.save_hyperparameters(arguments)
        num_labels = self.args.number_tag
        label2id_path = self.args.entity_label_file
        with open(label2id_path, 'r') as f:
            label2ids = json.load(f)
        self.id2label = {v: k for k, v in label2ids.items()}
        self.model = JDNerModel(arguments)
        # torch.save(self.bert.state_dict(), "data/init.pt")
        self.adv = arguments.adv
        if self.adv == "fgm":
            self.adv_model = FGM(self.model)
            # 加了对抗学习，要关闭LightningModule模块的自动优化功能
            self.automatic_optimization = False
        elif self.adv == 'pgd':
            self.adv_model = PGD(self.model)
            self.automatic_optimization = False

    def forward(self, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, labelids=None, max_sen_len=512):
        logit = self.model(input_ids, input_mask, pinyin_ids,
                           stroke_ids, max_sen_len)
        loss = self.global_pointer_crossentropy(labelids, logit)
        return loss

    def adv_fgm_model(self, fgm_model, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, labelids=None,  max_sen_len=512):
        # 对抗训练
        fgm_model.attack(epsilon=self.args.epsilon)  # 在embedding上添加对抗扰动
        # crf 训练
        logit = self.model(input_ids, input_mask, pinyin_ids,
                           stroke_ids, max_sen_len)
        loss_adv = self.global_pointer_crossentropy(labelids, logit)
        self.manual_backward(loss_adv)
        fgm_model.restore()
        return loss_adv.item()

    def adv_pgd_model(self, pgd_model, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, labelids=None, max_sen_len=512):
        pgd_model.backup_grad()
        K = self.args.pgd_K
        # 对抗训练
        loss_advs = []
        for t in range(K):
            # 在embedding上添加对抗扰动, first attack时备份param.data
            pgd_model.attack(epsilon=self.args.epsilon,
                             is_first_attack=(t == 0))
            if t != K - 1:
                self.model.zero_grad()
            else:
                pgd_model.restore_grad()

            logit = self.model(input_ids, input_mask, pinyin_ids,
                               stroke_ids, max_sen_len)
            loss_adv = self.global_pointer_crossentropy(labelids, logit)
            self.manual_backward(loss_adv)  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            loss_advs.append(loss_adv.item())
        pgd_model.restore()  #
        return np.array(loss_advs).mean()

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, pinyin_ids = batch['input_ids'], batch['input_mask'], batch['pinyin_ids']
        stroke_ids, lmask, labelids = batch['stroke_ids'], batch['lmask'], batch['pointer_labels']
        length = batch['length']
        max_sen_len = max(length)
        if not self.automatic_optimization:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            opt.zero_grad()

        # logit  = self.model(input_ids, input_mask,
        #                     pinyin_ids, stroke_ids, lmask, labelids)
        logit = self.model(input_ids, input_mask, max_sen_len=max_sen_len)
        loss = self.global_pointer_crossentropy(labelids, logit)
        if not self.automatic_optimization:
            self.manual_backward(loss)
            opt.step()
            scheduler.step()
        if self.adv == "fgm":
            self.adv_loss = self.adv_fgm_model(self.adv_model, input_ids, input_mask,
                                               labelids=labelids, max_sen_len=max_sen_len)
        elif self.adv == 'pgd':
            self.adv_loss = self.adv_pgd_model(self.adv_model, input_ids, input_mask,
                                               labelids=labelids, max_sen_len=max_sen_len)
        return loss

    def training_step_end(self, loss):
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def global_pointer_crossentropy(self, y_true, y_pred):
        """给GlobalPointer设计的交叉熵
        """
        # y_pred = (batch,l,l,c)
        bh = y_pred.shape[0] * y_pred.shape[1]
        y_true = torch.reshape(y_true, (bh, -1))
        y_pred = torch.reshape(y_pred, (bh, -1))
        return torch.mean(self.multilabel_categorical_crossentropy(y_true, y_pred))

    # 苏神的多标签分类损失，可以参加其硬截断损失那篇文章
    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def global_pointer_f1_score(self, y_true, y_pred):
        y_pred = torch.greater(y_pred, 0)
        # pre_index = y_pred.nonzero() #获取实体的索引[batch_index,type_index,start_index,end_index]
        # l = y_true * y_pred #预测正确的数量
        # h = y_true + y_pred #预测的数量+真实的数量
        return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()

    def on_validation_epoch_start(self) -> None:
        print('Valid.')

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, pinyin_ids = batch['input_ids'], batch['input_mask'], batch['pinyin_ids']
        stroke_ids, lmask, labelids = batch['stroke_ids'], batch['lmask'], batch['pointer_labels']
        length = batch['length']
        max_sen_len = max(length)
        # sequence_output = self.model.decode(input_ids=input_ids, attention_mask=input_mask, py2ids=pinyin_ids, sk2ids=stroke_ids)
        predict_tag = self.model(
            input_ids, input_mask, max_sen_len=max_sen_len)

        labelids = labelids.cpu().detach().numpy()
        predict_tag = predict_tag.cpu().detach().numpy()
        return (labelids, predict_tag)

    def validation_epoch_end(self, outputs) -> None:
        target_labels = []
        predict_labels = []
        for out in outputs:
            target_labels.append(out[0])
            predict_labels.append(out[1])
        char_acc = self.word_metric(target_labels, predict_labels)

        precision, recall, f1 = self.entity_metric(
            target_labels, predict_labels)
        self.log("char_acc", float(char_acc), prog_bar=True)
        self.log("pre", float(precision), prog_bar=True)
        self.log("recall", float(recall), prog_bar=True)
        self.log("f1", float(f1), prog_bar=True)

    def word_metric(self, target, predict):
        """字符级指标计算
        Args:
            target (_type_): _description_
            predict (_type_): _description_
        """
        TP = 0  # True Positive 预测为正例，实际为正例
        FP = 0  # False Positive 预测为正例，实际为负例
        TN = 0  # True Negative 预测为负例，实际为负例
        FN = 0  # False Negative 预测为负例，实际为正例
        all_words = 0
        for i in range(len(target)):
            tar = target[i]
            pre = predict[i]
            # 完全一致的预测数量
            TP += ((tar+pre) == 2).sum()
            all_words += (tar == 1).sum()
        acc = TP/all_words
        return acc

    def entity_metric(self, target, predict):
        """实体级的指标计算

        Args:
            target (_type_): _description_
            predict (_type_): _description_
        """
        gold_number = 0
        pred_number = 0
        correct_num = 0
        for i in range(len(target)):
            pre_index = np.greater(predict[i], 0)
            target_index = np.greater(target[i], 0)
            batch_size = target[i].shape[0]
            for j in range(batch_size):
                tar_entity = []
                for lab, start, end in zip(*(target_index[j].nonzero())):
                    tar_entity.append([self.id2label[lab], start, end])
                pre_entity = []
                for lab, start, end in zip(*(pre_index[j].nonzero())):
                    pre_entity.append([self.id2label[lab], start, end])
                pred_number += len(pre_entity)
                gold_number += len(tar_entity)
                for p_tag, p_start, p_end in pre_entity:
                    if any([p_tag == t_tag and p_start == t_start and p_end == t_end for t_tag, t_start, t_end in tar_entity]):
                        correct_num += 1

        precision = 0
        if pred_number != 0:
            precision = correct_num/pred_number
        recall = correct_num / gold_number
        f1 = 0
        if precision+recall != 0:
            f1 = 2*precision*recall/(precision+recall)
        return precision, recall, f1

    def build_entity(self, tags):
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

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        print('Test.')
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay) and "bert" in n], "weight_decay":0.8, "lr":self.args.lr},
            {"params": [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay) and "bert" in n], "weight_decay":0.0, "lr":self.args.lr},
            {"params": [p for n, p in param_optimizer if not any(
                [nd in n for nd in no_decay]) and 'bert' not in n], "weight_decay":0.8, 'lr':self.args.no_bert_lr},
            {"params": [p for n, p in param_optimizer if any(
                [nd in n for nd in no_decay]) and 'bert'not in n], 'weigth_decay':0.0, 'lr':self.args.no_bert_lr}
        ]
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in param_optimizer if "bert" in n],
        #         "weight_decay":0.8, "lr":self.args.lr},
        #     {"params": [p for n, p in param_optimizer if "bert" not in n],
        #         "weight_decay":0.0, "lr":self.args.no_bert_lr},
        # ]
        # optimizer_grouped_parameters = self.parameters()
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.args.lr)
        scheduler = LambdaLR(optimizer,
                             lr_lambda=lambda step: min((step + 1) ** -0.5,
                                                        (step + 1) * self.args.warmup_epochs ** (-1.5)),
                             last_epoch=-1)
        return [optimizer], [scheduler]

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.model.load_from_transformers_state_dict(gen_fp)
