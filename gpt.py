

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
data_file = 'data/input.txt' 
with open(data_file, 'r', encoding='utf-8') as f:
    text = f.read()



batch_size = 32
block_size = 8 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('torch cuda available', torch.cuda.is_available())
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)


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
print(data.shape, data.dtype)
print(data[:100])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  #print('ix shape:')
  #print(ix.shape)
  x = torch.stack([data[i:i+block_size] for i in ix])
  #print('x shape:')
  #print(x.shape)
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  #print('y shape:')
  #print(y.shape)
  return x, y

xb, yb = get_batch('train')
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
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


torch.manual_seed(1337)

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, C)
        q = self.query(x) #(B, T, C)
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        v = self.value(x) #(B, T, C)
        out = wei @ v #(B,T,T) @ ( B, T, C) -> (B, T, C)
        return out




class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
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


model = BigramLanguageModel(vocab_size)
m = model.to(device)

first_param = next(model.parameters())
print('model device',first_param.device)

first_param = next(m.parameters())
print('m device',first_param.device)

first_param = next(model.parameters())
logits, loss  = model(xb, yb)
print(logits.shape)
print(loss)




import time

# Start the timer
start_time = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32 
for step in range(max_iters):
    if step % eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# End the timer
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f'Time taken: {time_taken} seconds')


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
