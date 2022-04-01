"""
@Time   :   2021-01-12 15:08:01
@File   :   models.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import os
import json
from collections import OrderedDict, defaultdict
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel as bertModel, BertPooler, BertOnlyMLMHead, BertEmbeddings as bertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin
from torchcrf import CRF
from AdTraings import PGD, FGM
from lr_scheduler import CustomDecayLR, BertLR, ReduceLRWDOnPlateau, CosineLRWithRestarts, NoamLR
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau


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
    # def __init__(self, config):
    #     super().__init__(config)
    #     self.position_embeddings = SinusoidalPositionEmbedding(
    #         output_dim=config.hidden_size)

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
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # embeddings = self.position_embeddings(embeddings)
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


class JDNerModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = BertConfig.from_pretrained(self.args.bert_checkpoint)
        self.bert = BertModel(self.config, args)
        # self.bert = BertModel.from_pretrained(
        # self.args.bert_checkpoint, args=args)
        num_labels = self.args.number_tag
        self.add_lstm = self.args.lstm
        if self.add_lstm:
            self.is_bilstm = self.args.is_bilstm
            self.lstm_hidden = self.args.lstm_hidden
            self.lstm = nn.LSTM(self.config.hidden_size, self.lstm_hidden,
                                batch_first=True, bidirectional=self.is_bilstm, dropout=0.5)
            linear_hidden = self.lstm_hidden*2 if self.is_bilstm else self.lstm_hidden
        else:
            linear_hidden = self.config.hidden_size
        self.cls = nn.Linear(linear_hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, lmask=None, labelids=None, max_sen_len=512):
        sequence_output = self.bert(
            input_ids=input_ids, attention_mask=input_mask, py2ids=pinyin_ids, sk2ids=stroke_ids, max_sen_len=max_sen_len)
        if self.add_lstm:
            sequence_output, _ = self.lstm(sequence_output)
            sequence_output = sequence_output*input_mask.unsqueeze(-1)
        cls_output = self.cls(sequence_output)
        # crf 训练
        loss = -self.crf(cls_output, labelids, mask=lmask)
        return loss

    def decode(self, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, lmask=None, labelids=None, max_sen_len=None):
        sequence_output = self.bert(
            input_ids=input_ids, attention_mask=input_mask, py2ids=pinyin_ids, sk2ids=stroke_ids, max_sen_len=max_sen_len)
        if self.add_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        cls_output = self.cls(sequence_output)
        # crf 训练
        loss = 0
        if labelids is not None:
            loss = -self.crf(cls_output, labelids, mask=lmask)
        predict_tag = self.crf.decode(cls_output, mask=lmask)
        return predict_tag, loss

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


class CRFNerTrainingModel(pl.LightningModule):
    def __init__(self, arguments):
        super().__init__()
        self.args = arguments
        self.save_hyperparameters(arguments)
        label2id_path = self.args.label_file
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

    def forward(self, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, lmask=None, labelids=None, max_sen_len=512):
        loss = self.model(input_ids, input_mask, pinyin_ids,
                          stroke_ids, lmask, labelids, max_sen_len)
        return loss

    def adv_fgm_model(self, fgm_model, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, lmask=None, labelids=None, max_sen_len=512):
        # 对抗训练
        fgm_model.attack(epsilon=self.args.epsilon)  # 在embedding上添加对抗扰动
        # crf 训练
        loss_adv = self.model(input_ids, input_mask, pinyin_ids,
                              stroke_ids, lmask, labelids, max_sen_len)
        self.manual_backward(loss_adv)
        fgm_model.restore()
        return loss_adv.item()

    def adv_pgd_model(self, pgd_model, input_ids, input_mask, pinyin_ids=None, stroke_ids=None, lmask=None, labelids=None, max_sen_len=512):
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

            # crf 训练
            loss_adv = self.model(input_ids, input_mask, pinyin_ids,
                                  stroke_ids, lmask, labelids, max_sen_len)
            self.manual_backward(loss_adv)  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            loss_advs.append(loss_adv.item())
        pgd_model.restore()  #
        return np.array(loss_advs).mean()

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, pinyin_ids = batch['input_ids'], batch['input_mask'], batch['pinyin_ids']
        stroke_ids, lmask, labelids = batch['stroke_ids'], batch['lmask'], batch['labels']
        length = batch['length']
        max_sen_len = max(length)
        if not self.automatic_optimization:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            opt.zero_grad()

        loss = self.model(input_ids, input_mask,
                          lmask=lmask, labelids=labelids,)
        # loss = self.model(input_ids, input_mask, pinyin_ids=pinyin_ids, stroke_ids=stroke_ids,
        #                   lmask=lmask, labelids=labelids, max_sen_len=max_sen_len)
        if not self.automatic_optimization:
            self.manual_backward(loss)
            opt.step()
            scheduler.step()
        if self.adv == "fgm":
            self.adv_loss = self.adv_fgm_model(self.adv_model, input_ids, input_mask,
                                               lmask=lmask, labelids=labelids, max_sen_len=max_sen_len)
        elif self.adv == 'pgd':
            self.adv_loss = self.adv_pgd_model(self.adv_model, input_ids, input_mask,
                                               lmask=lmask, labelids=labelids, max_sen_len=max_sen_len)
        return loss

    def training_step_end(self, loss):
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, pinyin_ids = batch['input_ids'], batch['input_mask'], batch['pinyin_ids']
        stroke_ids, lmask, labelids = batch['stroke_ids'], batch['lmask'], batch['labels']
        length = batch['length']
        max_sen_len = max(length)
        predict_tag, val_loss = self.model.decode(
            input_ids, input_mask, lmask=lmask, labelids=labelids,)
        # predict_tag, val_loss = self.model.decode(input_ids, input_mask, pinyin_ids=pinyin_ids, stroke_ids=stroke_ids,
        #                                           lmask=lmask, labelids=labelids, max_sen_len=max_sen_len)
        val_loss = val_loss.cpu().detach().numpy()
        labelids = labelids.cpu().detach().numpy()
        return (labelids, predict_tag, val_loss, length)

    def on_validation_epoch_start(self) -> None:
        print('Valid.')

    def validation_epoch_end(self, outputs) -> None:
        target_labels = []
        predict_labels = []
        target_length = []
        loss = []
        for out in outputs:
            target_labels.extend(out[0])
            predict_labels.extend(out[1])
            loss.append(out[2])
            target_length.extend(out[3])
        loss = np.mean(loss)
        char_acc = self.word_metric(
            target_labels, predict_labels, target_length)

        precision, recall, f1 = self.entity_metric(
            target_labels, predict_labels, target_length)
        self.log("char_acc", float(char_acc), prog_bar=True)
        self.log("pre", float(precision), prog_bar=True)
        self.log("recall", float(recall), prog_bar=True)
        self.log("f1", float(f1), prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        # 主要是用于ReduceLRWDOnPlateau更新学习率的
        # res = pl.EvalResult(checkpoint_on=loss)

        # return res

    def word_metric(self, target, predict, length):
        """字符级指标计算
        Args:
            target (_type_): _description_
            predict (_type_): _description_
            length (_type_): _description_
        """
        TP = 0  # True Positive 预测为正例，实际为正例
        FP = 0  # False Positive 预测为正例，实际为负例
        TN = 0  # True Negative 预测为负例，实际为负例
        FN = 0  # False Negative 预测为负例，实际为正例
        all_words = 0
        for t, p, l in zip(*(target, predict, length)):
            t = t[:l]
            p = p[:l]
            # 完全一致的预测数量
            TP += sum([t1 == p1 for t1, p1 in zip(*(t, p))])
            FP += sum([t1 != p1 for t1, p1 in zip(*(t, p))])
            all_words += l

        acc = TP/all_words
        return acc

    def entity_metric(self, target, predict, length):
        """实体级的指标计算

        Args:
            target (_type_): _description_
            predict (_type_): _description_
            length (_type_): _description_
        """
        gold_number = 0
        pred_number = 0
        correct_num = 0
        each_entity = defaultdict(lambda: defaultdict(int))
        for t, p, l in zip(*(target, predict, length)):
            t = t[:l]
            l_tag = [self.id2label[line] for line in t]
            l_tags = self.build_entity(l_tag)
            for t, s, e in l_tags:
                each_entity[t]["glod_num"] += 1
            p = p[:l]
            p_tag = [self.id2label[line] for line in p]
            p_tags = self.build_entity(p_tag)
            pred_number += len(p_tags)
            gold_number += len(l_tags)
            for p_tag, p_start, p_end in p_tags:
                each_entity[p_tag]["pred_num"] += 1
                if any([p_tag == t_tag and p_start == t_start and p_end == t_end for t_tag, t_start, t_end in l_tags]):
                    correct_num += 1
                    each_entity[p_tag]["correct_num"] += 1
        precision = 0
        if pred_number != 0:
            precision = correct_num/pred_number
        recall = correct_num / gold_number
        f1 = 0
        if precision+recall != 0:
            f1 = 2*precision*recall/(precision+recall)
        self.show_each_entities(each_entity)
        return precision, recall, f1

    def show_each_entities(self, entities):
        """_summary_

        Args:
            entities (_type_): _description_
        """
        for lab, values in entities.items():
            if values['pred_num'] != 0:
                pre = values["correct_num"]/values['pred_num']
            else:
                pre = 0
            if values['glod_num'] != 0:
                rec = values["correct_num"]/values['glod_num']
            else:
                rec = 0
            if pre+rec != 0:
                f1 = 2*pre*rec/(pre+rec)
            else:
                f1 = 0
            print("标签:%3s 数量:%4d 预测数:%4d 准确数:%4d 准确率:%.5f 召回率:%.5f f1:%.5f" %
                  (lab, values['glod_num'], values['pred_num'], values['correct_num'], pre, rec, f1))

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
        # scheduler = LambdaLR(optimizer,
        #                      lr_lambda=lambda step: min((step + 1) ** -0.5,
        #                                                 (step + 1) * self.args.warmup_epochs ** (-1.5)),
        #                      last_epoch=-1)
        scheduler = ReduceLRWDOnPlateau(optimizer, min_lr=1e-9)
        scheduler = {
            'scheduler': scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.model.load_from_transformers_state_dict(gen_fp)
