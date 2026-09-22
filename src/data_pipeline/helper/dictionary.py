import torch
import duckdb

class HealthCareDictionary():
    def __init__(self, path = None):
        self.path = path
        self.str_to_id = {}
        self.id_to_str = {}
        if path:
            self.load(path)

    def __len__(self):
        return len(self.str_to_id)

    def add(self, tokens):
        if not isinstance(tokens, list):
            tokens = [tokens]
        for token in tokens:
            if token not in self.str_to_id:
                new_id = len(self.str_to_id)
                self.str_to_id[token] = new_id
                self.id_to_str[new_id] = token


    def encode(self, token):
        if isinstance(token, (list, tuple)):
            return [self.encode(t) for t in token]
        if token in self.str_to_id:
            return self.str_to_id[token]
        raise ValueError(f"Token '{token}' not found in vocabulary.")
    def decode(self, token_id):
        if isinstance(token_id, (list, tuple)):
            return [self.decode(tid) for tid in token_id]
        if token_id in self.id_to_str:
            return self.id_to_str[token_id]
        raise ValueError(f"Token ID '{token_id}' not found in vocabulary.")

    def save(self, path):
        torch.save({"str_to_id": self.str_to_id, "id_to_str": self.id_to_str}, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.str_to_id = checkpoint["str_to_id"]
        self.id_to_str = checkpoint["id_to_str"]
        