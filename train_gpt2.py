from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # 384 % 6 == 0 divide embedding dim by number of heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # new linear layer with input and 3x output
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # get output weighted based on attention
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # ensure model cannot see the future tokens
        # create a 2d tensor with 1s below the diagonal and 0s above -> 4D masking matrix(batch_size, attention_heads) (sequence positions, representing which tokens in the sequence are attending to which tokens)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding size
        
        qkv = self.c_attn(x) # linear projection. Each token emits 3 vectors, query, key, value
        q, k, v = qkv.split(self.n_embd, dim=2) # split the output into query, key, value
        
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
        return y

class MLP(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) 
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
    n_embd: int = 768

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__() 
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layer Nx
            ln_f = nn.LayerNorm(config.n_embd) # layernorm after the last block
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # classifier from embedding to vocab
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        
    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from hugging face """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained model {model_type}")
        
        # n_layer, n_head and n_embd are determined by model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600) # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPT2Config(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the attention bias

        # init hugging face transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard the attention masked bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard the attention bias
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"Length mismatch {len(sd_keys)} != {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T) # transpose
            else:
                # copy over the other parameters
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    

# ----------------------------------------------------------------------------------------------------------
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        with open('input.txt', 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens) 
        print(f"Total tokens {len(self.tokens)}")
        print(f"1 epoch has {len(self.tokens) // (B * T)} batches")
        
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B * T
        
        if self.current_position + B * T + 1 >= len(self.tokens):
            self.current_position = 0
        
        return x, y
# ----------------------------------------------------------------------------------------------------------
 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device {device}")

torch.manual_seed(1337)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)
elif torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# device = "cpu"
# get a data batch
train_loader = DataLoaderLite(B = 4, T = 32)


# model = GPT.from_pretrained('gpt2')
model = GPT(GPT2Config())
model.eval() # using not training
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step {i} loss {loss.item()}')

# print(loss)
import sys; sys.exit(0)

# prefix tokens

num_return_sequences = 5
max_length = 30

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long) # torch tensor
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # replicate 5 times
x = tokens.to(device)

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        # foward the model to get logits
        logits = model(x)
        logits = logits[:, -1, :] #(B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)
        
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
    

# stopped at 1:33