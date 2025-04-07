import string
from typing import Union
import math

from matplotlib import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules.container import T
# from adet.utils.queries import indices_to_text
import pytorch_lightning as pl
from timm.models.vision_transformer import VisionTransformer


CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
            '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', u'口']

ind_to_chr = {k: v for k, v in enumerate(CTLABELS)}
chr_to_ind = {v: k for k, v in enumerate(CTLABELS)}


space = [' ']
separator = [',', '-', '_']
special = ['!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
           '\\', ']', '^', '`', '{', '|', '}', '~']
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', u'口']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

query_space = torch.tensor([1 if c in space else 0 for c in CTLABELS] + [0])
query_separator = torch.tensor([1 if c in separator else 0 for c in CTLABELS] + [0])
query_special = torch.tensor([1 if c in special else 0 for c in CTLABELS] + [0])
query_alpha = torch.tensor([1 if c in alphabet else 0 for c in CTLABELS] + [0])
query_number = torch.tensor([1 if c in numbers else 0 for c in CTLABELS] + [0])
query_pad = torch.tensor([0 for _ in range(len(CTLABELS) + 1)])
query_pad[-1] = 1
query_empty = torch.tensor([0 for _ in range(len(CTLABELS) + 1)])


def compare_queries(query1, query2):
    return torch.all(torch.any(query1*query2, axis=1))


def max_query_types():
    return len(CTLABELS) + 1


def indices_to_text(indices):
    return "".join(ind_to_chr[index] if index != 96 else "" for index in indices)


def text_to_indices(text, pad=25):
    return [chr_to_ind[c] for c in text] + [96 for _ in range(pad - len(text))]




class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None
        self._emb_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def emb_key(self) -> str:
        return self._emb_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @emb_key.setter
    def emb_key(self, value: str):
        self._emb_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @emb_key.deleter
    def emb_key(self):
        del self._emb_key


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tile(self.pe[None, ...].to(x.device), (x.shape[0], 1, 1))
        return self.dropout(x)


class LabelEncoder(AbstractEmbModel, pl.LightningModule):

    def __init__(self, max_len, emb_dim, n_heads=8, n_trans_layers=12, ckpt_path=None, trainable=False, freeze=True,
                 lr=1e-4, lambda_cls=0.1, lambda_pos=0.1, clip_dim=1024, visual_len=197, visual_dim=768, visual_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_len = max_len
        self.emd_dim = emb_dim
        self.n_heads = n_heads
        self.n_trans_layers = n_trans_layers
        self.character = string.printable[:-6]
        # self.character = CTLABELS
        self.num_cls = len(self.character) + 1

        self.label_embedding = nn.Embedding(self.num_cls, self.emd_dim)
        self.pos_embedding = PositionalEncoding(d_model=self.emd_dim, max_len=self.max_len)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emd_dim, nhead=self.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_block, num_layers=self.n_trans_layers)
        # 2048--->256
        self.linear_projection = nn.Linear(2048, 256)
        
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"], strict=False)

        if freeze:    
            self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_index(self, labels):

        indexes = []
        for label in labels:
            assert len(label) <= self.max_len
            index = [self.character.find(c)+1 for c in label]
            index = index + [0] * (self.max_len - len(index))
            indexes.append(index)
        
        return torch.tensor(indexes, device=next(self.parameters()).device)
    
    def get_embeddings(self, x):
        
        emb = self.label_embedding(x)
        emb = self.pos_embedding(emb)
        out = self.encoder(emb)

        # linear_projection
        out = self.linear_projection(out)
        return out

    def forward(self, labels):
        texts = []
        for label in labels:
            label = label.tolist()
            for t in label: 
                text = indices_to_text(t[0])
                texts.append(text)
        idx = self.get_index(texts)
        out = self.get_embeddings(idx)

        return out