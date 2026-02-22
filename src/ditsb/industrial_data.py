"""
Industrial-Grade Data Loading

Wrappers for SOTA tokenizers (Tiktoken) to replace character-level datasets.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class TiktokenDataset(Dataset):
    """
    Dataset wrapper for OpenAI's tiktoken BPE tokenizer.
    
    Args:
        data (str): The raw text data.
        model_name (str): The encoding name (e.g., "gpt2", "cl100k_base").
        seq_len (int): Context window size.
    """
    def __init__(self, data: str, model_name: str = "gpt2", seq_len: int = 64):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("Please run 'pip install tiktoken' to use Industrial Data Loader.")
            
        self.tokenizer = tiktoken.get_encoding(model_name)
        # We encode the entire dataset into a numpy array of uint16 or int32
        self.tokens = np.array(self.tokenizer.encode(data), dtype=np.int32)
        self.seq_len = seq_len
        self.vocab_size = self.tokenizer.n_vocab
        
        print(f"TiktokenDataset: {len(self.tokens)} tokens, vocab={self.vocab_size}, encoding={model_name}")

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        # Return (input, target) where target is just the shifted input?
        # For DITSB flow matching, we just need a sequence x_1.
        chunk = self.tokens[idx : idx + self.seq_len]
        return torch.tensor(chunk, dtype=torch.long)

    def random_batch(self, batch_size: int, device: str) -> torch.Tensor:
        """
        Efficiently sample random chunks from the dataset.
        """
        # Random starting indices
        idx = torch.randint(0, len(self) - 1, (batch_size,))
        
        # Gather batch
        # This is faster than stacking __getitem__ calls for large batches
        batch = torch.stack([self[i.item()] for i in idx])
        return batch.to(device)

    def decode(self, tokens: torch.Tensor) -> list[str]:
        """
        Decodes a batch of token IDs back to strings.
        tokens: (B, L)
        """
        if tokens.dim() == 1:
            return [self.tokenizer.decode(tokens.cpu().numpy())]
        
        return [self.tokenizer.decode(seq.cpu().numpy()) for seq in tokens]
