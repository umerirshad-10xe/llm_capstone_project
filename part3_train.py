import torch
from config import max_iters, eval_interval, learning_rate, eval_iters, device
from part1_data import load_text, build_tokenizer, make_splits, get_batch
from part2_model import SingleHeadLanguageModel

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Compute average train/val loss over eval_iters batches.

    What you should do:
    - switch model to eval mode
    - loop `eval_iters` times and collect loss values
    - compute mean for train and val splits
    - switch model back to train mode
    - return a dict like: {"train": ..., "val": ...}
    """
    raise NotImplementedError

def main():
    """Train and then generate text.

    What you should do:
    - load text
    - build tokenizer (get vocab_size, encode, decode)
    - make train/val splits
    - initialize model + optimizer
    - run training loop:
        * periodically print train/val loss via estimate_loss()
        * sample training batch, forward, backward, optimizer step
    - generate a sample from a small context and print decoded output
    """
    raise NotImplementedError

if __name__ == "__main__":
    main()
