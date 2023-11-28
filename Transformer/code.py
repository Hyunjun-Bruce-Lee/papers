### Attention is all you need
### https://arxiv.org/abs/1706.03762

import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np
from copy import deepcopy as copy



### Common Layers
class ResidualConnectionLayer(nn.Module):
    def __init__(self, normalization, dr=0):
        super().__init__()
        self.normalization = normalization
        self.dropout = nn.Dropout(p=dr)

    def forward(self, x, sub_layer):
        out = x
        out = self.normalization(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out
    
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2, dr_rate=0):
        super().__init__()
        self.fc1 = fc1   
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2 

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy(qkv_fc)
        self.k_fc = copy(qkv_fc)
        self.v_fc = copy(qkv_fc)
        self.out_fc = out_fc              
        self.dropout = nn.Dropout(p=dr_rate)

    def calculate_attention(self, q, k, v, mask):
        d_k = k.shape[-1]
        attention_score = torch.matmul(q, k.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, v) # (n_batch, h, seq_len, d_k)
        return out

    def forward(self, q, k, v, mask=None):
        n_batch = q.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            out = out.transpose(1, 2)
            return out

        q = transform(q, self.q_fc)
        k = transform(k, self.k_fc) 
        v = transform(v, self.v_fc) 

        out = self.calculate_attention(q, k, v, mask) 
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.out_fc(out)
        return out










### Embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super().__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed, dr_rate=0):
        super().__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x):
        out = x
        out = self.embedding(out)
        out = self.dropout(out)
        return out




 






### Blocks
class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super().__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy(norm), dr_rate)

    def forward(self, src, src_mask):
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0):
        super().__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual3 = ResidualConnectionLayer(copy(norm), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(q=out, k=out, v=out, mask=tgt_mask))
        out = self.residual2(out, lambda out: self.cross_attention(q=out, k=encoder_out, v=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        return out
    







### Encoder
class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer, norm):
        super().__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)
        return out


### Decoder
class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer, norm):
        super().__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out)
        return out










### Transformer
class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    def make_pad_mask(self, query, key, pad_idx=1):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask

    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask










### Build model
def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("cpu"),
                max_len = 256,
                d_embed = 512,
                n_layer = 6,
                d_model = 512,
                h = 8,
                d_ff = 2048,
                dr_rate = 0.1,
                norm_eps = 1e-5):
    
    src_token_embed = TokenEmbedding(d_embed = d_embed,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(d_embed = d_embed,
                                     vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(d_embed = d_embed,
                                   max_len = max_len,
                                   device = device)
    src_embed = TransformerEmbedding(token_embed = src_token_embed,
                                     pos_embed = copy(pos_embed),
                                     dr_rate = dr_rate)
    tgt_embed = TransformerEmbedding(token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed),
                                     dr_rate = dr_rate)

    attention = MultiHeadAttentionLayer(d_model = d_model,
                                        h = h,
                                        qkv_fc = nn.Linear(d_embed, d_model),
                                        out_fc = nn.Linear(d_model, d_embed),
                                        dr_rate = dr_rate)
    position_ff = PositionWiseFeedForwardLayer(fc1 = nn.Linear(d_embed, d_ff),
                                               fc2 = nn.Linear(d_ff, d_embed),
                                               dr_rate = dr_rate)
    norm = nn.LayerNorm(d_embed, eps = norm_eps)
    encoder_block = EncoderBlock(self_attention = copy(attention),
                                 position_ff = copy(position_ff),
                                 norm = copy(norm),
                                 dr_rate = dr_rate)
    decoder_block = DecoderBlock(self_attention = copy(attention),
                                 cross_attention = copy(attention),
                                 position_ff = copy(position_ff),
                                 norm = copy(norm),
                                 dr_rate = dr_rate)
    encoder = Encoder(encoder_block = encoder_block,
                      n_layer = n_layer,
                      norm = copy(norm))
    decoder = Decoder(decoder_block = decoder_block,
                      n_layer = n_layer,
                      norm = copy(norm))
    generator = nn.Linear(d_model, tgt_vocab_size)
    model = Transformer(src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device
    return model