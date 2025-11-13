# data.py - build vocab, encode text, provide get_batch()
import os, pickle
import torch

def build_dataset(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    text = ""
    raw_lines = []
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            raw_lines.extend(lines)
            text += "\n".join(lines) + "\n"
    
    # Original character-level vocab
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    
    # Convert text to tensor
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # --- New: build character labels ---
    characters = []
    char_to_idx = {}
    idx_to_char = {}
    char_text_pairs = []

    for line in raw_lines:
        if ':' in line:
            char, sentence = line.split(':', 1)
            char = char.strip()
            sentence = sentence.strip()
            char_text_pairs.append((char, sentence))
            if char not in characters:
                characters.append(char)
    
    char_to_idx = {c:i for i,c in enumerate(characters)}
    idx_to_char = {i:c for i,c in enumerate(characters)}
    
    meta = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(chars),
        "characters": characters,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "char_text_pairs": char_text_pairs  # optional, for batch generation
    }
    
    return train_data, val_data, meta

# --- Updated get_batch to explicitly pass char_to_idx ---
def get_batch(data, batch_size, block_size, device, char_text_pairs=None, char_to_idx=None):
    ix = torch.randint(0, len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    
    # If character info is provided, pick random characters for the batch
    char_idx = None
    if char_text_pairs is not None and char_to_idx is not None:
        char_samples = [char_text_pairs[i % len(char_text_pairs)][0] for i in ix]
        # Convert characters to indices
        char_idx = torch.tensor([char_to_idx[c] for c in char_samples], device=device)
    
    return (x, y) if char_idx is None else (x, y, char_idx)

def save_meta(meta, path):
    with open(path, "wb") as f:
        pickle.dump(meta, f)

def load_meta(path):
    with open(path, "rb") as f:
        return pickle.load(f)
