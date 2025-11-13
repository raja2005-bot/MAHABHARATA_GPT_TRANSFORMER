# train.py - orchestrates training
import argparse, os, time
import torch
from data import build_dataset, get_batch, save_meta
from model import GPTMini

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=7500)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    print("Building dataset...")
    train_data, val_data, meta = build_dataset(args.data_dir)
    args.meta = meta
    save_meta(meta, os.path.join(args.save_dir, "meta.pkl"))
    print("Vocab size:", meta["vocab_size"])
    print("Number of characters:", len(meta["characters"]))

    # --- Updated: Pass n_chars to model ---
    model = GPTMini(
        meta["vocab_size"], 
        args.block_size, 
        n_chars=len(meta["characters"])
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for it in range(args.max_iters):
        model.train()
        # --- Updated: Always pass char_to_idx ---
        xb, yb, char_idx = get_batch(
            train_data,
            args.batch_size,
            args.block_size,
            device,
            meta["char_text_pairs"],
            meta["char_to_idx"]   # <-- added this
        )
        logits, loss = model(xb, yb, char_idx=char_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb, char_idx = get_batch(
                    train_data,
                    args.batch_size,
                    args.block_size,
                    device,
                    meta["char_text_pairs"],
                    meta["char_to_idx"]   # <-- added this
                )
                _, val_loss = model(xb, yb, char_idx=char_idx)
            print(f"Step {it}: train loss {loss.item():.4f} -> val loss {val_loss.item():.4f}")
            # save checkpoint
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "meta": meta
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_{it}.pt"))

if __name__ == "__main__":
    main()
