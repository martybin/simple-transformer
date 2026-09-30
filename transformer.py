import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2
DIGIT_OFFSET = 3
VOCAB = DIGIT_OFFSET + 10

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                        # x: (B, T, d_model)
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def _split(self, x):                         # (B,T,D) -> (B,h,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        B, Tq, D = x_q.shape
        q, k, v = self._split(self.w_q(x_q)), self._split(self.w_k(x_kv)), self._split(self.w_v(x_kv))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)          # (B,h,Tq,Tk)
        if mask is not None:                                             # True = Access
            scores = scores.masked_fill(~mask, -1e9)
        attn = scores.softmax(dim=-1)
        self.attn = attn.detach()
        out = self.dropout(attn) @ v                                     # (B,h,Tq,d_k)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.w_o(out)
        # we can also replace this: F.scaled_dot_product_attention(q, k, v, attn_mask=mask), with this function (FlashAttention).


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, src_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, tgt_mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, src_mask))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=4, d_ff=128, n_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.enc_norm, self.dec_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, emb, tokens):
        return self.drop(self.pos(emb(tokens) * math.sqrt(self.d_model)))

    def encode(self, src, src_mask):
        x = self.embed(self.src_emb, src)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.embed(self.tgt_emb, tgt)
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.generator(self.dec_norm(x))

    def forward(self, src, tgt):
        src_mask, tgt_mask = make_masks(src, tgt)
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)


def make_masks(src, tgt):
    src_mask = (src != PAD)[:, None, None, :]
    T = tgt.size(1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
    tgt_mask = (tgt != PAD)[:, None, None, :] & causal
    return src_mask, tgt_mask


def make_batch(batch_size, device, min_len=3, max_len=8):
    src = torch.full((batch_size, max_len), PAD)
    tgt_in = torch.full((batch_size, max_len + 1), PAD)
    tgt_out = torch.full((batch_size, max_len + 1), PAD)
    for b in range(batch_size):
        n = torch.randint(min_len, max_len + 1, (1,)).item()
        digits = torch.randint(0, 10, (n,))
        s = digits.sort().values
        src[b, :n] = digits + DIGIT_OFFSET
        tgt_in[b, 0] = BOS
        tgt_in[b, 1:n + 1] = s + DIGIT_OFFSET
        tgt_out[b, :n] = s + DIGIT_OFFSET
        tgt_out[b, n] = EOS
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


@torch.no_grad()
def greedy_decode(model, src, max_len=10):
    src_mask = (src != PAD)[:, None, None, :]
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), BOS, device=src.device)
    for _ in range(max_len):
        _, tgt_mask = make_masks(src, ys)
        nxt = model.decode(ys, memory, src_mask, tgt_mask)[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if (ys == EOS).any(1).all():
            break
    return ys[:, 1:]


def to_digits(seq):
    out = []
    for t in seq.tolist():
        if t in (EOS, PAD):
            break
        out.append(t - DIGIT_OFFSET)
    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, warmup, steps = 64, 200, 1500

    model = Transformer(VOCAB, d_model=d_model, n_heads=4, d_ff=128, n_layers=2, dropout=0.0).to(device)
    print(f"device={device}  parameters={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    noam = lambda s: 0.5 * d_model ** -0.5 * min((s + 1) ** -0.5, (s + 1) * warmup ** -1.5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam)

    model.train()
    for step in range(1, steps + 1):
        src, tin, tout = make_batch(64, device)
        logits = model(src, tin)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tout.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.5f}")

    model.eval()
    src, _, _ = make_batch(500, device)
    pred = greedy_decode(model, src)
    acc = sum(to_digits(p) == sorted(to_digits(s)) for p, s in zip(pred, src)) / 5
    print(f"\nexact-match accuracy: {acc:.1f}%")
    for i in range(5):
        print(f"  {to_digits(src[i])}  ->  {to_digits(pred[i])}")

    torch.save(model.state_dict(), "sorter_transformer.pt")
