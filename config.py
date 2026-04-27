import torch

# ======================
# Hyperparameters (edit)
# ======================
batch_size = 16
block_size = 64
max_iters = 5000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 200

# Model size
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0

# Part 4 (GQA) knobs
# - n_kv_head must divide n_head (e.g., 4/2, 8/2, 8/4)
n_kv_head = 2

# MLP / modern block knobs
ff_mult = 4  # expansion factor for MLP hidden size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
