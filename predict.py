import sys
import torch
from transformer import Transformer, greedy_decode, to_digits, DIGIT_OFFSET

CKPT = sys.argv[1] if len(sys.argv) > 1 else "sorter_transformer.pt"
N_HEADS = 4
MIN_LEN = 3
MAX_LEN = 8

state = torch.load(CKPT, map_location="cpu")
vocab, d_model = state["src_emb.weight"].shape
d_ff = state["enc_layers.0.ffn.net.0.weight"].shape[0]
n_layers = len({k.split(".")[1] for k in state if k.startswith("enc_layers.")})
MAX_NUM = vocab - DIGIT_OFFSET

model = Transformer(vocab, d_model=d_model, n_heads=N_HEADS, d_ff=d_ff,
                    n_layers=n_layers, dropout=0.0)
model.load_state_dict(state)
model.eval()

print(f"model: {CKPT}  (vocab={vocab}, d_model={d_model}, d_ff={d_ff}, layers={n_layers})")
print(f"Enter {MIN_LEN} to {MAX_LEN} numbers between 0 and {MAX_NUM - 1}, separated by spaces. Exit: q")

while True:
    text = input(">>> ").strip()
    if text.lower() == "q":
        break
    if not text:
        continue
    try:
        nums = [int(x) for x in text.split()]
    except ValueError:
        print("Only enter the integer numbers")
        continue
    if not all(0 <= n < MAX_NUM for n in nums):
        print(f"Each number must be between 0 and {MAX_NUM - 1}.")
        continue
    if not MIN_LEN <= len(nums) <= MAX_LEN:
        print(f"The number of digits must be between {MIN_LEN} and {MAX_LEN}.")
        continue

    src = torch.tensor([[n + DIGIT_OFFSET for n in nums]])
    pred = to_digits(greedy_decode(model, src, max_len=MAX_LEN + 2)[0])
    print(f"    {pred}  {'✓' if pred == sorted(nums) else '✗'}")
