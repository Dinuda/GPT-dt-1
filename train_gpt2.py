from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # 384 % 6 == 0 divide embedding dim by number of heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # new linear layer with input and 3x output
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # get output weighted based on attention
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # ensure model cannot see the future tokens
        # create a 2d tensor with 1s below the diagonal and 0s above -> 4D masking matrix(batch_size, attention_heads) (sequence positions, representing which tokens in the sequence are attending to which tokens)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding size
        
        qkv = self.c_attn(x) # linear projection. Each token emits 3 vectors, query, key, value
        q, k, v = qkv.chunk(3, dim=-1) # split into query, key, value
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, self.n_head, C // self.n_head) -> (B, self.n_head, T, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, self.n_head, C // self.n_head) -> (B, self.n_head, T, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, self.n_head, C // self.n_head) -> (B, self.n_head, T, C // self.n_head)
        
        # attention matrix - materialize the attention matrix of T, T for query, key
        # k.transpose(-2, -1) -> swap the last two dimensions
        # scale the scores - divide by sqrt of the key size to stabilize the gradients, prev from becoming too large
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # mask out the future tokens
        att = F.softmax(att, dim=-1) # softmax over the last dimension
        y = att @ v # multiply the attention matrix with the value matrix - weighted sum of the values
        y = y.transpose(1, 2).contiguous().view(B, T, C) # transpose back and reshape - concat   
        # output projection
        y = self.c_proj(y)

class MLP(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # self.attn = CasualSelfAttention(config) # TODO:
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # forward pass
        x = x + self.attn(self.ln_1(x)) # reduce the impact of the positional encoding
        x = x + self.mlp(self.ln_2(x)) # feedforward - residual connection
        return x

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 786

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__() 
        self.config = config
        
        self.transformer - nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layer Nx
            ln_f = nn.LayerNorm(config.n_embd) # layernorm after the last block
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # classifier from embedding to vocab