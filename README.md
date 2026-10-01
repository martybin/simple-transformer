# Simple Transformer — from scratch

A minimal, fully commented **Encoder–Decoder Transformer** ("Attention Is All You Need", Vaswani et al., 2017) written in PyTorch, trained on a toy sequence-to-sequence task: **sorting a sequence of digits**.

```
input:   7 2 9 2 5 0
output:  0 2 2 5 7 9
```

There is no sorting algorithm anywhere in the code — the model learns the task purely from examples. The point of the project is to have a small, readable Transformer that you can understand line by line and then extend to real tasks (translation, OCR, time series, …).

## Results

| Setting | Value |
|---|---|
| Parameters | 170,189 |
| Layers | 2 encoder + 2 decoder |
| `d_model` / heads / `d_ff` | 64 / 4 / 128 |
| Training | 1,500 steps, batch 64, ~1 min on CPU |
| Exact-match accuracy (500 unseen sequences) | **99.6 %** |

An *exact match* means the whole output sequence must be correct.

## Files

| File | Purpose |
|---|---|
| `transformer.py` | Model, data generator, training loop, evaluation. Saves `sorter_transformer.pt`. |
| `predict.py` | Loads the trained weights and sorts numbers you type in. |
| `transformer_numpy.py` | *(optional)* The same model in pure NumPy, including a hand-written autograd engine and a gradient check — for learning what PyTorch does under the hood. |

## Quick start

```bash
pip install torch            # numpy + matplotlib only for the NumPy version

python transformer.py        # trains (~1 min on CPU) and writes sorter_transformer.pt
python predict.py            # interactive inference
```

```
>>> 9 4 4 1 0 7
    [0, 1, 4, 4, 7, 9]  ✓
>>> q
```

`predict.py` reads `d_model`, `d_ff`, vocabulary size and number of layers directly from the checkpoint, so it stays in sync with whatever you trained. To load another checkpoint: `python predict.py my_model.pt`.

## How it works

### Vocabulary
```
PAD = 0    BOS = 1    EOS = 2    digit d → token d + 3   (vocab size 13)
```

### Data
For each example the decoder input and the target are shifted by one position (*teacher forcing*):
```
source          : 3 1 4 1 5
decoder input   : BOS 1 1 3 4 5
target          : 1 1 3 4 5 EOS
```
Sequences have random lengths (3–8) and are padded with `PAD`; padded positions are masked in attention and ignored in the loss.

### Architecture
```
src → Embedding·√d + PositionalEncoding → [Encoder layer × N] → memory
tgt → Embedding·√d + PositionalEncoding → [Decoder layer × N] → Linear → logits
                                               ↑ cross-attention on memory
```

- **Multi-head attention** — `softmax(QKᵀ/√d_k + mask)·V`, computed in `h` parallel heads of size `d_model/h`.
- **Encoder layer** — self-attention → feed-forward.
- **Decoder layer** — masked (causal) self-attention → cross-attention over the encoder output → feed-forward.
- **Pre-LayerNorm residual blocks** — `x = x + Sublayer(LayerNorm(x))`, plus a final LayerNorm after each stack (more stable than the original Post-LN).
- **Masks** — a padding mask on keys, and a causal (lower-triangular) mask in the decoder so position *i* cannot see the future.
- **Sinusoidal positional encoding** — attention is order-agnostic, so positions are added explicitly.

### Training
- Adam (β = 0.9, 0.98), Noam learning-rate schedule (linear warm-up for 200 steps, then ∝ 1/√step).
- Cross-entropy with `ignore_index=PAD`, gradient clipping at 1.0.

### Inference
Greedy autoregressive decoding: encode once, then start from `BOS` and repeatedly append the most likely next token until `EOS`.

## Customizing

Only the **data** and the **vocabulary** need to change — the architecture stays the same.

| Goal | Change |
|---|---|
| Reverse instead of sort | in `make_batch`: `s = digits.flip(0)` |
| Numbers 0–99 instead of digits | `MAX_NUM = 100`, `VOCAB = DIGIT_OFFSET + MAX_NUM`, `torch.randint(0, MAX_NUM, …)`; use a bigger model (`d_model=128`) and more steps |
| Longer sequences | `max_len` in `make_batch`, `max_len` in `greedy_decode`, `MAX_LEN` in `predict.py` |
| Bigger model | `d_model`, `n_heads`, `d_ff`, `n_layers` (keep `d_model % n_heads == 0`) |

The model only handles inputs similar to its training data: tokens it has never seen don't exist in its vocabulary, and much longer sequences than it was trained on usually fail.

## Troubleshooting

| Error | Cause / fix |
|---|---|
| `IndexError: index out of range in self` | An input token is outside the vocabulary (e.g. the number `12` for a digits-only model). Retrain with a larger `MAX_NUM`, or keep inputs in range. |
| `FileNotFoundError: sorter_transformer.pt` | Run `python transformer.py` first, in the same folder; check that it ends with `torch.save(...)`. |
| `size mismatch` when loading | The model in `predict.py` was built with different settings from the checkpoint — use the provided `predict.py`, which reads them from the file. `n_heads` must still match. |

## Where to go next

The same building blocks give every well-known Transformer family:

- **GPT (decoder-only)** — drop the encoder and cross-attention; train on next-token prediction.
- **BERT (encoder-only)** — encoder layers without the causal mask + a classification head on a `[CLS]` token.
- **ViT** — split an image into 16×16 patches, project each patch to `d_model`, feed the encoder.
- **Modern upgrades** — `F.scaled_dot_product_attention` (FlashAttention), RoPE, RMSNorm, SwiGLU, KV cache, beam search.

The follow-up project in this series turns this model into an **English → Persian translator** (subword tokenizer, real parallel data, beam search, BLEU).

## Reference

Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)