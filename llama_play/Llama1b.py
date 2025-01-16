import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Llama1b:
    vocab_size = 128256
    layers = 16
    attn_heads = 16
    n_embd = 2048


class LlamaModel(nn.Module):

    def __init__(self, config:dataclass):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            self_attn = nn.ModuleList([LlamaAttention(config) for _ in range(config.layers)]),
            norm = LlamaRMSNorm((2048,), eps=1e-5),
            rotary_emb = LlamaRotaryEmbeddings(),
        ))

        self.lm_head = nn.Embedding(config.n_embd, config.vocab_size, bias=False)


class LlamaAttention(nn.Module):

    def __init__(self, config:dataclass):
        super().__init__()
        self.config = config


class LlamaMLP(nn.Module):

    def __init__(self, config:dataclass):
        super().__init__()
        self.config = config


class LlamaRMSNorm(nn.Module):

    def __init__(self, dim:tuple, lr:float):
        super().__init__()


class LlamaRotaryEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()