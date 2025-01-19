import torch
import torch.nn as nn
from typing import Tuple, Optional
from configs import LlamaConfig, llama_3_2_1b

# View the configs.py file to understand this configuration abstraction
# essentially, this is where all of the parameters for initializing
# the model size and rope embeddings
config = LlamaConfig(**llama_3_2_1b)

################################
## Llama Model Implementation ##
################################
class LlamaModel(nn.Module):

    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            self_attn = nn.ModuleList([LlamaDecoderBlock(config) for _ in range(config.layers)]),
            norm = LlamaRMSNorm((2048,), eps=1e-5),
            rotary_emb = LlamaRotaryEmbedding(),
        ))

        self.lm_head = nn.Embedding(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, x):
        ...

#########################
## Llama Decoder Block ##
#########################
class LlamaDecoderBlock(nn.Module):
    ...

####################################
## Llama Feed Forward Layer (MLP) ##
####################################
class LlamaMLP(nn.Module):

    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

#############
## RMSNorm ##
#############
class LlamaRMSNorm(nn.Module):
    """
    Root Mean Square Norm
    """
    def __init__(self, hidden_size:torch.tensor, eps:float=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

######################
## Rotary Embedding ##
######################

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
    def __init__(self, config:LlamaConfig):
        super().__init__()

        # BC: "rope_type" was originally "type"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        
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
