import torch
from transformer import Transformer, greedy_decode, to_digits, VOCAB, DIGIT_OFFSET

model = Transformer(VOCAB, d_model=64, n_heads=4, d_ff=128, n_layers=2, dropout=0.0)
model.load_state_dict(torch.load("sorter_transformer.pt", map_location="cpu"))
model.eval()

print("Enter the numbers with a space and for quit enter q: ")
while True:
    text = input(">>> ").strip()
    if text == "q":
        break
    digits = [int(x) for x in text.split()]
    src = torch.tensor([[d + DIGIT_OFFSET for d in digits]])
    pred = to_digits(greedy_decode(model, src)[0])
    ok = "✓" if pred == sorted(digits) else "✗"
    print(f"    {pred}  {ok}")
