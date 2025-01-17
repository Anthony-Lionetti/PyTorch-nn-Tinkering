import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Llama1b:
    block_size = 2048
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
    """
    Root Mean Square Norm
    ---
    Differs from the traditional transformer Architecture's LayerNorm.
    RMS is computationally more efficient, but slightly less accurate. 
    In practice, the perfromance differences are negligible.
    """

    def __init__(self, hidden_size:tuple, eps:1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states:torch.Tensor):
        # convert hidden states to float32
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # Square the terms, and calculate the mean
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return the normalized weights
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()