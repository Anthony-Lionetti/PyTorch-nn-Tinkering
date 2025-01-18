import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    block_size = 2_048
    vocab_size = 128_256
    n_embd = 2_048
    intermediate_size = 8_192
    layers = 16
    attn_heads = 32
    rope_theta = 500_000
    attention_dropout = 0.0
    attention_bias = False
    mlp_bias = False


class LlamaModel(nn.Module):

    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            self_attn = nn.ModuleList([LlamaAttention(config) for _ in range(config.layers)]),
            norm = LlamaRMSNorm((2048,), eps=1e-5),
            rotary_emb = LlamaRotaryEmbedding(),
        ))

        self.lm_head = nn.Embedding(config.n_embd, config.vocab_size, bias=False)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Gets the head dimension from the config, or returns a default head_dim calculation
        self.head_dim = getattr(config, "head_dim", config.n_embd // config.attn_heads)
        self.num_key_value_groups = config.attn_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.n_embd, config.attn_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.n_embd, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.n_embd, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.attn_heads * self.head_dim, config.n_embd, bias=config.attention_bias)

    def forward(self,x: torch.Tensor, position_embeddings:tuple[torch.Tensor, torch.Tensor]):
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(x).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class LlamaMLP(nn.Module):
    # TODO: Sort out sizes and fix LLamaConfig accordingly

    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

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

############################
# Helper Functions for RoPE #
############################

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.max_seq_len = max_position_embeddings
        self.dim = dim
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            position_ids: Tensor of shape (batch_size, seq_len) containing position indices
        """
        # Create frequency matrix
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies for each position
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        
        # Create rotation matrix components
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
