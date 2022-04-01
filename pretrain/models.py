"""
@Time   :   2021-01-12 15:08:01
@File   :   models.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import os
import json
from collections import OrderedDict
import torch
from torch import logit, nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertEmbeddings as bertEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):
    def __init__(self, config, PYLEN, num_embeddings, max_sen_len=512):
        super().__init__()
        self.config = config
        self.PYDIM = 30
        self.seq_len = PYLEN
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

    def forward(self, sen_pyids,max_sen_len):
        sen_pyids = sen_pyids.reshape(-1, self.seq_len)
        sen_emb = self.pyemb(sen_pyids)
        sen_emb = sen_emb.reshape(-1, self.seq_len, self.PYDIM)
        all_out, final_out = self.gru(sen_emb)
        final_out = final_out.mean(0, keepdim=True)
        lstm_output = final_out.reshape(
            shape=[-1, max_sen_len, self.config.hidden_size])

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
    def __init__(self, config, args):
        super().__init__()
        PYLEN, SKLEN = args.pylen, args.sklen
        self.config = config
        self.pyemb = EmbeddingNetwork(
            self.config, PYLEN=PYLEN, num_embeddings=30)
        self.skemb = EmbeddingNetwork(
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
            py_emb = self.pyemb(py2ids,max_sen_len)
            pinyin_emb = py_emb
        if sk2ids is not None:
            sk_emb = self.skemb(sk2ids,max_sen_len)
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

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = torch.load(gen_fp)
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            # if name.startswith('encoder'):
            #     name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)


class JDNerTrainingModel(pl.LightningModule):
    def __init__(self, arguments):
        super().__init__()
        self.args = arguments
        self.save_hyperparameters(arguments)
        label2id_path = self.args.label_file
        with open(label2id_path, 'r') as f:
            label2ids = json.load(f)
        self.id2label = {v: k for k, v in label2ids.items()}
        self.config = BertConfig.from_pretrained(self.args.bert_checkpoint)
        self.bert = BertModel(self.config, arguments)
        # bert_static = torch.load("lightning_logs/plome/checkpoints/bert.pt",map_location="cpu")
        # self.bert.load_state_dict(bert_static)
        self.cls = nn.Linear(self.config.hidden_size, arguments.num_labels)
        # cls_static = torch.load("lightning_logs/plome/checkpoints/cls.pt",map_location="cpu")
        # self.cls.load_state_dict(cls_static)
        self.py_cls = nn.Linear(self.config.hidden_size,
                                arguments.py_num_labels)
        # _py_cls_static = torch.load("lightning_logs/plome/checkpoints/py_cls.pt",map_location="cpu")
        # self.py_cls.load_state_dict(_py_cls_static)            
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)
        self.py_loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        self.min_loss = float('inf')

    def forward(self, input_ids, input_mask, masked_pinyin_ids=None, masked_sk_ids=None, pinyin_ids=None, lmask=None, labelids=None,max_sen_len=512):
        sequence_output = self.bert(
            input_ids=input_ids, attention_mask=input_mask, py2ids=None, sk2ids=None,max_sen_len=max_sen_len)
        lmask = ~lmask.type(torch.bool)
        labelids = torch.masked_fill(labelids, lmask, torch.tensor(-100))
        labelids = labelids.reshape(-1)
        # pinyin_ids = torch.masked_fill(
        #     pinyin_ids, lmask, torch.tensor(-100))
        # pinyin_ids = pinyin_ids.reshape(-1)
        logits = self.cls(sequence_output)
        logits = logits.reshape(-1, self.args.num_labels)
        loss = self.loss_function(logits, labelids)
        # py_logit = self.py_cls(sequence_output)
        # py_logit = py_logit.reshape(-1, self.args.py_num_labels)
        # py_loss = self.py_loss_function(py_logit, pinyin_ids)
        return {"loss":loss}

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, pinyin_ids = batch['input_ids'], batch['input_mask'], batch['pinyin_ids']
        masked_pinyin_ids, lmask, labelids = batch['masked_pinyin_ids'], batch['lmask'], batch['labels']
        masked_sk_ids = batch['masked_sk_ids']
        max_sen_len = max(batch['length'])
        # loss = self.forward(input_ids, input_mask, masked_pinyin_ids, masked_sk_ids,
        #                     pinyin_ids, lmask, labelids,max_sen_len)
        loss = self.forward(input_ids, input_mask, lmask= lmask, labelids=labelids,max_sen_len=max_sen_len)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, pinyin_ids = batch['input_ids'], batch['input_mask'], batch['pinyin_ids']
        masked_pinyin_ids, lmask, labelids = batch['masked_pinyin_ids'], batch['lmask'], batch['labels']
        masked_sk_ids = batch['masked_sk_ids']
        max_sen_len = max(batch['lengthes'])
        # loss = self.forward(input_ids, input_mask, masked_pinyin_ids, masked_sk_ids,
        #                     pinyin_ids, lmask, labelids,max_sen_len)
        loss = self.forward(input_ids, input_mask,lmask= lmask, labelids=labelids,max_sen_len=max_sen_len)
        return loss.cpu().numpy()

    def on_validation_epoch_start(self) -> None:
        print('Valid.')

    def validation_epoch_end(self, outputs) -> None:
        loss = []
        for out in outputs:
            loss.append(out)
        loss = np.mean(loss)
        self.log("val_loss", loss)
        # return loss

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

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.bert.load_from_transformers_state_dict(gen_fp)
