import torch
import torch.nn as nn
from typing_extensions import Tuple
from configs import LlamaConfig, llama_3_2_1b
import math

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
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config        

        # Create inverse frequency bands
        inv_freq = self._rope_init_fn(config=config)
        self.inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        

    def _rope_init_fn(self, config: LlamaConfig):
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))

        factor = config.rope_scaling["factor"]  # `8` in the original implementation
        low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
        high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
        old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation
       
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

        
    @torch.no_grad()
    def forward(self, x:torch.Tensor, position_ids:torch.Tensor):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


#####################
## Llama Attention ##
#####################

def rotate_half(x:torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q:torch.Tensor, k:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaAttention(nn.Module):
    """Multi-headed Attention"""

    def __init__(self, config:LlamaConfig, layer_idx:int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        ## Query Projection: transforms input embeddings into query vectors (What "I am" interested in)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        ## Key Projection: transforms input embeddings into key vectors (Here is what "I" have)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        ## Value Projection: transforms input embeddings into value vectors (If you "find me interesting" here is what I communicate)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        ## Output Projection: concatenates the outputs from all attention heads, projecting them back to the original head size
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    
    def forward(self, hidden_states:torch.Tensor, position_embeddings:Tuple[torch.Tensor, torch.Tensor]):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)