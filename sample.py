# sample.py - load checkpoint and generate text
import argparse, torch
from data import load_meta
from model import GPTMini
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None, char_idx=None):
    for _ in range(max_new_tokens):
        # Only feed the last block_size tokens to the model
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond, char_idx=char_idx)

        # Focus only on the last token's logits
        logits = logits[:, -1, :] / temperature

        # Top-k filtering (optional)
        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Turn into probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Sample from distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append sampled token
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--start", type=str, default="Arjuna: ")
    parser.add_argument("--character", type=str, default="Arjuna")  # --- New: character input
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=64)   # <--- match training block_size
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    # Build model with correct vocab, block size, and number of characters
    model = GPTMini(meta["vocab_size"], block_size=args.block_size, n_chars=len(meta["characters"])).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Character mappings
    stoi = meta["stoi"]
    itos = meta["itos"]
    char_to_idx = meta["char_to_idx"]

    # Convert start text to tensor
    start_ids = torch.tensor([[stoi.get(ch, 0) for ch in args.start]], device=args.device)
    # Get character index
    char_idx = torch.tensor([char_to_idx.get(args.character, 0)], device=args.device)

    # Generate text
    out = generate(
        model,
        start_ids,
        max_new_tokens=args.length,
        block_size=args.block_size,  # dynamically use same block size
        temperature=args.temperature,
        top_k=args.top_k,
        char_idx=char_idx
    )

    # Decode output
    print("".join([itos[int(i)] for i in out[0].tolist()]))


if __name__ == "__main__":
    main()
