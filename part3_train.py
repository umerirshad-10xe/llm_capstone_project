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
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
    text = load_text()
    vocab_size, stoi, itos, encode, decode = build_tokenizer(text)
    train_data, val_data = make_splits(text, encode)
    model = SingleHeadLanguageModel(vocab_size)
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

if __name__ == "__main__":
    main()
