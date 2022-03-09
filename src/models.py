"""
@Time   :   2021-01-12 15:08:01
@File   :   models.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import operator
import os
from collections import OrderedDict
from tkinter.messagebox import NO

import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertModel, BertEmbeddings as bertEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin

from .utils import compute_corrector_prf, compute_sentence_level_prf
import numpy as np


class EmbeddingNetwork(nn.Module):
    def __init__(self, config, PYLEN, num_embeddings, max_sen_len=512):
        super().__init__()
        self.config = config
        self.PYDIM = 30
        self.seq_len = PYLEN
        num_embeddings = num_embeddings
        self.pyemb = nn.Embedding(num_embeddings, self.PYDIM)
        self.gru = nn.GRU(
            self.PYDIM,
            self.config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True,
        )
        self.MAX_SEN_LEN = max_sen_len

    def forward(self, sen_pyids):
        sen_pyids = sen_pyids.reshape(-1, self.seq_len)
        sen_emb = self.pyemb(sen_pyids)
        sen_emb = sen_emb.reshape(-1, self.seq_len, self.PYDIM)
        all_out, final_out = self.gru(sen_emb)
        lstm_output = final_out.reshape(
            shape=[-1, self.MAX_SEN_LEN, self.config.hidden_size])

        return lstm_output


class BertEmbeddings(bertEmbeddings):
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
        if pinyin_embs is not None:
            embeddings += pinyin_embs
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, device):
        super().__init__()
        self.config = config
        self.pyemb = EmbeddingNetwork(self.config, PYLEN=4, num_embeddings=30)
        self.skemb = EmbeddingNetwork(self.config, PYLEN=10, num_embeddings=7)
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pooler = BertPooler(self.config)
        self.cls = BertOnlyMLMHead(self.config)
        self._device = device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        return_dict=None,
        py2ids=None,
        sk2ids=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        py_emb = self.pyemb(py2ids)
        sk_emb = self.skemb(sk2ids)
        pinyin_emb = py_emb + sk_emb
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            pinyin_embs=pinyin_emb)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = torch.load(gen_fp)
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)


class BaseCorrectorTrainingModel(pl.LightningModule):
    """
    用于CSC的BaseModel, 定义了训练及预测步骤
    """

    def __init__(self, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = arguments
        self.w = arguments.loss_weight
        self.min_loss = float('inf')

    def training_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            det_acc_labels.append(
                det_predict[1:len(_src) + 1].equal(det_label[1:len(_src) + 1]))
            results.append((_src, _tgt, _predict,))

        return loss.cpu().item(), det_acc_labels, cor_acc_labels, results

    def on_validation_epoch_start(self) -> None:
        print('Valid.')

    def validation_epoch_end(self, outputs) -> None:
        det_acc_labels = []
        cor_acc_labels = []
        results = []
        for out in outputs:
            det_acc_labels += out[1]
            cor_acc_labels += out[2]
            results += out[3]
        loss = np.mean([out[0] for out in outputs])
        print(f'loss: {loss}')
        print(f'Detection:\n'
              f'acc: {np.mean(det_acc_labels):.4f}')
        print(f'Correction:\n'
              f'acc: {np.mean(cor_acc_labels):.4f}')
        print('Char Level:')
        compute_corrector_prf(results)
        compute_sentence_level_prf(results)
        if (len(outputs) > 5) and (loss < self.min_loss):
            self.min_loss = loss
            torch.save(self.state_dict(),
                       os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))
            print('model saved.')
        torch.save(self.state_dict(),
                   os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        print('Test.')
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(optimizer,
                             lr_lambda=lambda step: min((step + 1) ** -0.5,
                                                        (step + 1) * self.args.warmup_epochs ** (-1.5)),
                             last_epoch=-1)
        return [optimizer], [scheduler]


class SoftMaskedBertModel(BaseCorrectorTrainingModel):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args = args
        self.tokenizer = tokenizer
        self.corrector = BertModel(self.config, tokenizer, args.device)
        self._device = args.device

    def forward(self, texts, cor_labels=None, det_labels=None):
        encoded_texts = self.tokenizer(
            texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],
                                          token_type_ids=encoded_texts['token_type_ids'])
        prob = self.detector(embed)
        cor_out = self.corrector(
            texts, prob, embed, cor_labels, residual_connection=True)

        if det_labels is not None:
            det_loss_fct = nn.BCELoss(reduction='sum')
            # pad部分不计算损失
            active_loss = encoded_texts['attention_mask'].view(
                -1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            outputs = (det_loss, cor_out[0], prob.squeeze(-1)) + cor_out[1:]
        else:
            outputs = (prob.squeeze(-1),) + cor_out

        return outputs

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.corrector.load_from_transformers_state_dict(gen_fp)
