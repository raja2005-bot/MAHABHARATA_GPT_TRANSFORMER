✨ Project Overview

Mahabharata GPT is a character-level transformer model that generates authentic Mahabharata-style dialogues for different characters like Draupadi, Krishna, and Arjuna.

It’s trained on custom .txt datasets and learns both character speech patterns and sequences, enabling context-aware text generation.

Key Features:

Character-level dialogue generation

Context-aware character embeddings

Checkpoint saving for resuming or testing at any stage

GPU-optimized training with PyTorch

📂 Project Structure
maha_project/
│
├─ data/                  # Text files of Mahabharata dialogues
├─ src/
│   ├─ data.py            # Prepares dataset, vocab, batching
│   ├─ model.py           # GPTMini transformer model
│   ├─ train.py           # Training loop, optimizer, checkpoint saving
│   └─ sample.py          # Generate dialogues from trained model
├─ checkpoints/           # Saved model checkpoints & meta.pkl
├─ requirements.txt       # All dependencies
└─ README.md              # Project documentation

⚡ Live Demo (Optional)

Generate dialogues in style of Mahabharata using trained checkpoints:

!python src/sample.py \
  --ckpt checkpoints/ckpt_2000.pt \
  --start "Draupadi: " \
  --length 300 \
  --temperature 0.7 \
  --top_k 50 \
  --block_size 128 \
  --device cuda


Example output:

Draupadi: O Krishna, why do the Pandavas hesitate in the court?  
Krishna: Fear not, Draupadi. The dharma will guide us through.  
Arjuna: I shall obey the path of righteousness.

🛠️ Installation

Clone the repo:

git clone https://github.com/yourusername/mahabharata-gpt.git
cd mahabharata-gpt


Install dependencies:

pip install -r requirements.txt


Enable GPU in Colab or local machine for faster training:

Runtime → Change runtime type → GPU

🏋️ Training
!python src/train.py \
  --data_dir data \
  --save_dir checkpoints \
  --batch_size 16 \
  --block_size 128 \
  --max_iters 2000 \
  --eval_interval 200 \
  --lr 3e-4 \
  --device cuda


Parameters explained:

Parameter	Description
batch_size	Number of sequences processed per step
block_size	Length of each input sequence (context)
max_iters	Total training iterations
eval_interval	Steps after which validation and checkpoints are saved
device	cuda for GPU, cpu if GPU unavailable

Checkpoints: Saved in checkpoints/ as ckpt_XXXX.pt along with meta.pkl.

🔮 Inference / Text Generation
!python src/sample.py \
  --ckpt checkpoints/ckpt_2000.pt \
  --start "Krishna: " \
  --length 300 \
  --temperature 0.7 \
  --top_k 50 \
  --block_size 128 \
  --device cuda


Parameters:

--start → Prompt text

--length → Number of characters to generate

--temperature → 0.1 = safe, 1.0 = creative

--top_k → Restrict prediction to top-k most probable characters

🔗 Workflow
graph LR
A[Raw Text Files] --> B[data.py: dataset, vocab, batching]
B --> C[train.py: training loop + checkpoint saving]
C --> D[model.py: GPTMini transformer model]
C --> E[checkpoints: saved models + meta]
E --> F[sample.py: text generation]

🌟 Future Improvements

Increase dataset for better context coherence

Larger block size and batch size for longer sequences

Word-level GPT for more human-like sentences

Attention-based character embeddings for consistent dialogue

📚 References

PyTorch Documentation

GPT Paper – Attention Is All You Need

Character-level Language Models – Karpathy
