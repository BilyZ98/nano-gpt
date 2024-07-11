
import os

import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
dir = '/GPUFS/nsccgz_qylin_1/zt/persona_txt/'
data_file = dir + '/knowledge.txt'
with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()

# batch_size = 32
# block_size = 8 # what is the maximum context length for predictions
# max_iters = 5000
# eval_interval = 500
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('torch cuda available', torch.cuda.is_available())
# eval_iters = 200
# n_embd = 32
# n_head = 4
# n_layer = 3
# dropout = 0.2

torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------
# simple lauch: python persona_gpt.py
# DDP launch :
# torchrun --standalone  -nproc_per_node=8 persona_gpt.py
# 
start_time = time.time()
ddp = int(os.environ.get('RANK',-1)) != -1
if ddp:
    assert torch.cuda.is_available(), " for now I think we need cuda for ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    print(device)
    print('ddp_local_rank', ddp_local_rank)
    print('ddp_rank', ddp_rank)
    print('ddp_world_size', ddp_world_size)
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('torch cuda available', torch.cuda.is_available())

batch_size = 64
block_size = 256 # what is the maximum context length for predictions
max_iters = 50
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

print("I am gpu:", ddp_rank)
# import sys; sys.exit(0)

device_type = "cuda" if device.startswith("cuda") else "cpu"


chars = sorted(list(set(text)))
print('chars = ', chars)
vocab_size = len(chars)
print('vocab_size = ', vocab_size)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
print('Load tokens:', data.shape, data.dtype)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
if device_type == "cuda":
    torch.cuda.manual_seed(1337)



def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

class DataLoader:
    def __init__(self, data, B, T,  process_rank=0, num_process=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        self.current_idx = self.B * self.T * self.process_rank
        self.data = data

    def __len__(self):
        return len(self.data)

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.data[self.current_idx:self.current_idx + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_idx += B * T * self.num_process
        if self.current_idx + B * T * self.num_process + 1> len(self.data):
            self.current_idx = self.B * self.T * self.process_rank
        return x, y

data_loader = DataLoader(train_data, batch_size, block_size, ddp_rank, ddp_world_size )
eval_data_loader = DataLoader(val_data, batch_size, block_size, ddp_rank, ddp_world_size )

xb, yb = data_loader.next_batch()
print('xb shape:')
print(xb.shape)
print(xb)
print('yb shape:')
print(yb.shape)
print(yb)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, C)
        q = self.query(x) #(B, T, C)
        v = self.value(x) #(B, T, C)

        # wei = q @ k.transpose(-2, -1) * C **-0.5
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # wei = F.softmax(wei, dim=-1) # (B, T, T)
        # wei = self.dropout(wei)
        # out = wei @ v #(B,T,T) @ ( B, T, C) -> (B, T, C)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return out



class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd*4),
                nn.ReLU(),
                nn.Linear(4*n_embd, n_embd),
                nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transfomer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa = Head(n_embd)
    # self.sa_heads = MultiHeadAttention(4, n_embd//4)
    # self.ffw = FeedForward(n_embd)
    # self.blocks = nn.Sequential(
    #         Block(n_embd, n_head=4),
    #         Block(n_embd, n_head=4),
    #         Block(n_embd, n_head=4),
    #         nn.LayerNorm(n_embd),
    # )
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.token_embedding_table.weight = self.lm_head.weight

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    # x = self.sa(x)  #(B,T, C)
    # x = self.sa_heads(x) #(B, T, C)
    # x = self.ffw(x) #(B, T, C)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
        

    loss = None
    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T) if targets is not None else None
        loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      #print('shape of logits', logits.shape)
      logits = logits[:, -1, :] # becomes (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx

def generate_tokens(model, idx, max_new_tokens, yb=None):
    model.eval()
    #with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    idx = idx.to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            print('idx length:', len(idx))
            idx_cond = idx[:, -block_size:]
            print('idx_cond length:', len(idx_cond))
            idx_cond_device = idx_cond.device
            print('idx_cond device', idx_cond_device)
            B, T = idx_cond.shape
            print('idx_cond shape:', B, T)
            logits, loss = model(idx_cond, yb) # (B,T,vocab_size)
            #print('shape of logits', logits.shape)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    model.train()
    return idx

def call_generate_tokens(model):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device )
    # cur_text = "hi"
    # idx = torch.tensor(encode(cur_text), dtype=torch.long, device=device)
    idx_device = idx.device
    print('idx device:', idx_device)
    # print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
    start_time = time.time()
    generated_idx = generate_tokens(model, idx, 32 )[0].tolist()
    end_time = time.time()
    print('token generation time:', end_time-start_time)
    start_time = time.time()
    decode_txt = decode(generated_idx)
    print(f"rank: {ddp_local_rank}, {decode_txt}")
    end_time = time.time()
    print('token decode time', end_time-start_time)


model = BigramLanguageModel(vocab_size)
m = model.to(device)
use_compile = False
if use_compile:
    start_time = time.time()
    model = torch.compile(model)
    end_time = time.time()
    print('compile time:', end_time-start_time)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

first_param = next(model.parameters())
print('model device',first_param.device)

first_param = next(m.parameters())
print('m device',first_param.device)

end_time = time.time()
print('ini time:', end_time - start_time)


# Start the timer
start_time = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(max_iters):
    t0 = time.time()
    # xb, yb = get_batch('train')
    xb, yb = data_loader.next_batch()
    xb, yb = xb.to(device), yb.to(device)
    B, T = xb.shape
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss = model(xb, yb)

    # if ddp:
    #     model.require_backward_grad_sync = True
    loss.backward()

    if ddp:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    token_processed = B * T * ddp_world_size
    #if step % eval_interval == 0 :
    #    losses = estimate_loss()
    #    print(f"step {iter}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}")

    dt = (t1 - t0) * 1000 # milli sec
    token_per_sec = token_processed/ (t1-t0)
    if master_process:
        call_generate_tokens(model)
        print(f'step {step}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {token_per_sec:.2f}')
        


# End the timer
end_time = time.time()


# Calculate and print the time taken
time_taken = end_time - start_time

if master_process:
    print(f'Time taken: {time_taken} seconds')

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    # Assuming 'model' is your PyTorch model
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


    

if ddp:
    destroy_process_group()


