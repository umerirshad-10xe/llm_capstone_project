import csv
import torch

import config as cfg
from part1_data import load_text, build_tokenizer, make_splits, get_batch
from part4_models import MHA_ModernLanguageModel, GQA_ModernLanguageModel

# ==========================================
# Part 4D: Experiments + learning summary
# ==========================================
# Goal:
# - Keep cfg.n_embd fixed
# - Try different head counts, layers, and learning rates
# - Compare MHA vs GQA (and different cfg.n_kv_head for GQA)
# - Record validation loss and write a small summary table
#
# Suggestions:
# - Start with short runs (few hundred to a couple thousand steps)
# - Change one variable at a time
# - Be careful with constraints:
#     * cfg.n_embd % cfg.n_head == 0
#     * cfg.n_head % cfg.n_kv_head == 0

@torch.no_grad()
def eval_val_loss(model, val_data, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    dummy_train = val_data
    for k in range(eval_iters):
        X, Y = get_batch("val", dummy_train, val_data)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train_for_steps(model, train_data, val_data, steps, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

def main():
    # TODO:
    # 1) Load text + tokenizer + train/val splits
    # 2) Choose sweeps for:
    #    - cfg.n_head (ensure divisibility)
    #    - cfg.n_layer
    #    - learning rate
    #    - cfg.n_kv_head (for GQA, must divide cfg.n_head)
    # 3) For each setting:
    #    - set cfg.n_head, cfg.n_layer, cfg.n_kv_head
    #    - build model (MHA_ModernLanguageModel or GQA_ModernLanguageModel)
    #    - train short steps
    #    - record validation loss
    # 4) Write:
    #    - CSV with all runs
    #    - Markdown table with best runs and a short “learnings” section
    raise NotImplementedError

if __name__ == "__main__":
    main()
