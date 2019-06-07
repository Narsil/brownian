import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.decoder = TiedLMHead(self.embedding.weight)
        attn_block = AttentionBlock(
            config.EMBEDDING_DIM, config.CONTEXT_SIZE, config.N_HEAD
        )
        self.encoder = nn.Sequential(
            *[copy.deepcopy(attn_block) for _ in range(config.N_BLOCKS)]
        )

    def forward(self, indices):
        embedded = self.embedding(indices)  # embedded: [B, C, D]
        attn = self.encoder(embedded)  # attn: [B, C, D]
        return attn


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def generate_tokens_from_indices(self, indices, add_tokens=1):
    context = indices  # [1, C, K]
    new_tokens = []
    with torch.no_grad():
        self.eval()
        for i in range(add_tokens):
            attn = self.forward(context)
            decoded = self.decoder(attn)  # decoded: [1, C, V]
            logits = F.log_softmax(decoded, -1)
            batch_tokens = logits.argmax(dim=2)
            new_context = batch_tokens
            new_context[:, :-1] = context[:, 1:]  # Stabilise
            context = new_context
            new_tokens.append(new_context[:, -1:])
    self.train()
    all_tokens = torch.cat([indices[:, :], *new_tokens], dim=1)
    return all_tokens.squeeze(0)


def gelu(x):
    """Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415
    """
    a = np.sqrt(2 / np.pi) * (x + 0.044_715 * torch.pow(x, 3))
    return 0.5 * x * (1 + torch.tanh(a))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        """2 layer feed forward neural net

        Args:
            dim_in: int
            dim_out: int
        """
        super(MLP, self).__init__()
        self.ffn_hidden = Linear(dim_out, dim_in)
        self.ffn_out = Linear(dim_in, dim_out)
        self.act = gelu

    def forward(self, x):
        """
        Args:
            x: FloatTensor[B, C, D]
        Returns:
            FloatTensor[B, C, D]
        """
        h = self.act(self.ffn_hidden(x))
        h2 = self.ffn_out(h)
        return h2


class AttentionBlock(nn.Module):
    def __init__(self, dim, csz, n_head):
        """Base blocks for a transformer, which is k of those chained"""
        super(AttentionBlock, self).__init__()

        self.attn = MultiHeadAttention(dim, csz, n_head)
        self.ln_1 = LayerNorm(dim)
        self.mlp = MLP(dim, 4 * dim)
        self.ln_2 = LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: FloatTensor[B, C, D]
        """
        attn = self.attn(x)  # attn: [B, C, D]
        normed = self.ln_1(x + attn)  # normed: [B, C, D]
        projed = self.mlp(normed)  # projed: [B, C, D]
        out = self.ln_2(normed + projed)  # out: [B, C, D]
        return out


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Linear(nn.Module):
    def __init__(self, out_dim, in_dim):
        super(Linear, self).__init__()

        self.out_dim = out_dim

        w = torch.empty(in_dim, out_dim)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: [B, C, D]
        size_out = x.size()[:-1] + (self.out_dim,)
        x = torch.addmm(
            self.b, x.view(-1, x.size(-1)), self.w
        )  # isn't that just nn.Linear?
        x = x.view(*size_out)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, csz, n_head):
        """Multi head attention as described by Vaswani 2017

        Args:
            dim             : int -> input dimension
            csz             : int -> max context size
            n_head          : int -> number of heads (must divide the dimension)
        """
        super(MultiHeadAttention, self).__init__()

        assert dim % n_head == 0
        self.register_buffer(  # ~ parameter without gradient
            "mask", torch.tril(torch.ones(csz, csz)).view(1, 1, csz, csz)
        )

        self.n_head = n_head
        self.dim = dim

        self.ffn_make_qkv = Linear(dim * 3, dim)
        self.ffn_out = Linear(dim, dim)

    def _attn(self, q, k, v):
        """Perform the attention in the canonical form:
            attn(query, key, values) = sum_i softmax(query.dot(key))_i values_i
        Args:
            query: FloatTensor[B, H, C, D//H]
            key  : FloatTensor[B, H, D//H, C]
            value: FloatTensor[B, H, C, D//H]
        """
        w = torch.matmul(q, k)  # w: [B, H, C, C]
        w = w / (v.size(-1) ** 0.5)

        mask = self.mask[:, :, : w.size(-2), : w.size(-1)]  # mask: [1, 1, C, C]
        w = w * mask + -1e9 * (1 - mask)  # w: [B, H, C, C]

        w = nn.Softmax(dim=-1)(w)  # w: [B, H, C, C]
        attn = torch.matmul(w, v)  # attn: [B, H, C, D//H]
        return attn

    def merge_heads(self, x):
        """Merge the heads: need to reshape the tensor and view it properly
        Args:
            x: FloatTensor[B, H, C, D//H]
        Returns:
            merged: FloatTensor[B, C, D]
        """
        b, h, c, dh = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # x: [B, C, H, D//H]
        merged = x.view(b, c, h * dh)
        return merged

    def split_heads(self, x, is_key=False):
        """Split along the dimension in n_heads
        Args:
            x: FloatTensor[B, C, D]
            is_key: bool -> wether to transpose to allow matmul in _attn
        Returns:
            is_key=false -> FloatTensor[B, H, C, D//H]
            is_key=true  -> FloatTensor[B, H, D//H, C]
        """
        b, c, d = x.size()
        split_shape = (b, c, self.n_head, d // self.n_head)
        x = x.view(*split_shape)  # x: [B, C, H, D//H]
        if is_key:
            return x.permute(0, 2, 3, 1)  # x: [B, H, D//H, C]
        else:
            return x.permute(0, 2, 1, 3)  # x: [B, H, C, D//H]

    def forward(self, x):
        """Performs multi-head attention. Tensor shapes described with:
        B: batch_size, C: context_size, D: dimension, H: number of heads

        Args:
            x: FloatTensor[B, C, D]
        Returns:
            a: FloatTensor[B, C, D]
        """

        x = self.ffn_make_qkv(x)  # x: [B, C, 3D]
        query, key, value = x.split(self.dim, dim=2)  # each: [B, C, D]

        query = self.split_heads(query)  # query: [B, H, C, D//H]
        key = self.split_heads(key, is_key=True)  # key: [B, H, D//H, C]
        value = self.split_heads(value)  # value: [B, H, C, D//H]

        attn = self._attn(query, key, value)  # attn: [B, H, D//H, D//H]
        merged = self.merge_heads(attn)  # a: [B, C, D]
        out = self.ffn_out(merged)  # a: [B, C, D]
        return out


class TiedLMHead(nn.Module):
    def __init__(self, embedding):
        """Simple decoder, but ties the output matrix to the embedding
        as advised by: https://arxiv.org/pdf/1608.05859.pdf

        Args:
            embedding: FloatTensor[V, D] ->
        """
        super(TiedLMHead, self).__init__()

        vocab_size, dim = embedding.size()
        self.decoder = nn.Linear(dim, vocab_size, bias=False)
        self.decoder.weight = embedding

    def forward(self, h):
        """
        Args:
            h: FloatTensor[B, C, D]
        Returns:
            FloatTensor[B, C, V]
        """
        return self.decoder(h)
