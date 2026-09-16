class HealtcareVocab(nn.Module):
    def __init__(self):
        
class HealthcareVocab(nn.Module):
    def __init__(self, special_tokens=None, vocab_map=None, index_to_token=None, table_masks=None):
        super().__init__()
        self.special_tokens = special_tokens or {"<PAD>": 0, "<UNK>": 1, "<NULL>": 2, "<NaT>": 3}
        self.vocab_map = vocab_map or {}
        self.index_to_token = index_to_token or [{"index": idx, "table": None, "column": None, "value": tok} for tok, idx in self.special_tokens.items()]

        # Register table masks as PyTorch buffers for automatic GPU/CPU transfer
        if table_masks:
            for table_name, mask_tensor in table_masks.items():
                self.register_buffer(f"mask_{table_name}", mask_tensor)

    @property
    def vocab_size(self):
        return len(self.index_to_token)

    def get_table_mask(self, table_name):
        """Returns the boolean tensor mask for a given table."""
        return getattr(self, f"mask_{table_name}")

    def encode_value(self, table):


    def encode_value(self, table, column, value):
        """Maps a (table, column, value) string triplet to a token ID."""
        if value is None:
            return self.special_tokens["<NULL>"]
        if str(value) in self.special_tokens.keys():
            return self.special_tokens[str(value)]
        return self.vocab_map.get((table, column, str(value)), self.special_tokens["<UNK>"])
    def encode(self, table, column, values):
        """Tokenizes a batch of string values into a 1D PyTorch LongTensor."""
        ids = [self.encode_value(table, column, v) for v in values]
        return torch.tensor(ids, dtype=torch.long)
    def decode(self, token_ids):
        """Decodes token IDs back to their metadata (table, column, value)."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [
            self.index_to_token[tid] if 0 <= tid < len(self.index_to_token) else {"index": tid, "table": None, "column": None, "value": "<UNK>"}
            for tid in token_ids
        ]
    def save(self, path):
        """Saves the PyTorch module state and vocabulary metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "state_dict": self.state_dict(),
            "special_tokens": self.special_tokens,
            "vocab_map": self.vocab_map,
            "index_to_token": self.index_to_token
        }
        torch.save(save_dict, path)
    @classmethod
    def load(cls, path):
        """Loads a saved vocabulary checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        vocab = cls(special_tokens=checkpoint["special_tokens"], vocab_map=checkpoint["vocab_map"], index_to_token=checkpoint["index_to_token"])

        # Pre-register mask buffers stored in state_dict so PyTorch expects them
        state_dict = checkpoint["state_dict"]
        for key, tensor in state_dict.items():
            if key.startswith("mask_") and not hasattr(vocab, key):
                vocab.register_buffer(key, torch.zeros_like(tensor))

        vocab.load_state_dict(state_dict)
        return vocab
    

