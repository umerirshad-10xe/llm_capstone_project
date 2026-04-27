import torch
from config import batch_size, block_size, device

def load_text(path: str = "input.txt") -> str:
    """Read the dataset file and return it as a single Python string.

    What you should do:
    - Open the file at `path` using UTF-8 encoding.
    - Read the entire contents and return it.
    """
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError

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
    raise NotImplementedError
