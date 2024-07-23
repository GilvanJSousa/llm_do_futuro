import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
dropout = 0.2


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_ff, 4 * n_embd_ff),
            nn.ReLU(),
            nn.Linear(4 * n_embd_ff, n_embd_ff),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd_block, n_head_block):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd_block // n_head_block
        self.sa = MultiHeadAttention(n_head_block, head_size)
        self.ffwd = FeedFoward(n_embd_block)
        self.ln1 = nn.LayerNorm(n_embd_block)
        self.ln2 = nn.LayerNorm(n_embd_block)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


def read_and_create_variable(file_path_read):
    unique_lines_var = set()

    try:
        with open(file_path_read, 'r') as file:
            for line in file:
                unique_lines_var.add(line.strip())
    except FileNotFoundError:
        print(f"File not found: {file_path_read}")
        return None

    return sorted(list(unique_lines_var))


file_path = 'logs/game_grid_log.txt'  # Replace with the path to your text file
unique_lines = read_and_create_variable(file_path)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.readlines()

vocab_size = len(unique_lines)
n_embd = 8
n_head = 1
n_layer = 1
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head_block=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


PATH = "models/state_dict_model.pt"
model = BigramLanguageModel()
model.load_state_dict(torch.load(PATH))
m = model.to(device)

stoi = {ch: i for i, ch in enumerate(unique_lines)}
itos = {i: ch for i, ch in enumerate(unique_lines)}
encode = lambda s: [stoi[c[1:-1]] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: '\n'.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

print(stoi)
print(itos)

exemplo = [7]

context = torch.tensor([exemplo], dtype=torch.long, device=device)
#context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("context", context[0][0])
resp = m.generate(context, max_new_tokens=3)[0].tolist()
print(resp)
print(decode(resp))
#print(decode(m.generate(context, max_new_tokens=3)[0].tolist()))
