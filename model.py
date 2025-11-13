# model.py - minimal decoder-only transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=384, n_head=6, n_layer=6, dropout=0.2, n_chars=None, char_emb_size=32):
        super().__init__()
        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # --- New: Character embeddings ---
        self.char_emb = None
        if n_chars is not None:
            self.char_emb = nn.Embedding(n_chars, char_emb_size)
            # Optional linear to project concatenated embedding back to model dim
            self.char_proj = nn.Linear(n_embd + char_emb_size, n_embd)
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # Output head
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, char_idx=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)
        x = tok
        # --- Add character embedding if provided ---
        if self.char_emb is not None and char_idx is not None:
            char_emb = self.char_emb(char_idx)[:, None, :].expand(-1, T, -1)
            x = torch.cat([tok, char_emb], dim=-1)
            x = self.char_proj(x)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(x + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
