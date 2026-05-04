import torch
from config import batch_size, block_size, device

def load_text(path: str = "input.txt") -> str:
    """Read the dataset file and return it as a single Python string.

    What you should do:
    - Open the file at `path` using UTF-8 encoding.
    - Read the entire contents and return it.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_tokenizer(text: str):
    """Build a character-level tokenizer.

    Return:
      - vocab_size: number of unique characters
      - stoi: dict mapping character -> integer id
      - itos: dict mapping integer id -> character
      - encode_fn(s): converts a string into a list[int] token ids
      - decode_fn(ids): converts list[int] token ids back into a string

    What you should do:
    - Collect the unique characters appearing in `text` (order them deterministically).
    - Create forward/backward lookup tables (stoi/itos).
    - Implement encode/decode functions using those lookups.
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return vocab_size, stoi, itos, encode, decode

def make_splits(text: str, encode_fn):
    """Numericalize the dataset and make train/val splits.

    Return:
      - train_data: 1D LongTensor of token ids (first ~90%)
      - val_data:   1D LongTensor of token ids (last ~10%)

    What you should do:
    - Convert the full text into token ids using `encode_fn`.
    - Wrap into a torch.LongTensor.
    - Split by position: first 90% train, remaining 10% validation.
    """
    data = torch.tensor(encode_fn(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(split: str, train_data, val_data):
    """Sample a batch for next-token prediction.

    Return:
      - x: (B, T) LongTensor of token ids
      - y: (B, T) LongTensor of target ids (x shifted by one position)

    What you should do:
    - Choose `data` based on split.
    - Randomly sample `batch_size` starting positions such that a full block fits.
    - Build x sequences of length `block_size`.
    - Build y sequences as the *next token* labels for each position in x.
    - Move both tensors to `device`.
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
