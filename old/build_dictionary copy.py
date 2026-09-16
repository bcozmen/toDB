from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb
import torch
import torch.nn as nn

DATASET_PATH = Path("/home/baris/database_transformer/dataset")
DATABASE_PATH = DATASET_PATH / "healthcare.duckdb"
DEFAULT_OUTPUT = DATASET_PATH / "ml" / "vocab.pt"

VOCAB_COLUMNS = {
    "patients": {
        "gender": "gender",
        "race": "race",
        "ethnicity": "ethnicity",
        "birthplace": "birthplace",
        "city": "city",
    },
    "encounters": {"event_type": "encounter", "code": "code"},
    "conditions": {"event_type": "condition", "code": "code"},
    "medications": {"event_type": "medication", "code": "code"},
    "procedures": {"event_type": "procedure", "code": "code"},
    "observations": {
        "event_type": "observation_numeric",
        "code": "code",
        "units": "units",
    },
    "immunizations": {"event_type": "immunization", "code": "code"},
    "allergies": {"event_type": "allergy", "code": "code"},
    "careplans": {"event_type": "careplan", "code": "code"},
    "imaging_studies": {"event_type": "imaging", "code": "bodysite_code"},
}


class HealthcareVocab(nn.Module):
    """PyTorch-native reversible, table/column-aware vocabulary."""

    def __init__(
        self,
        special_tokens: Optional[Dict[str, int]] = None,
        vocab_map: Optional[Dict[tuple[str, str, str], int]] = None,
        index_to_token: Optional[List[dict]] = None,
        table_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.special_tokens = special_tokens or {"<PAD>": 0, "<UNK>": 1, "<NULL>": 2, "<NaT>": 3}
        self.vocab_map = vocab_map or {}
        self.index_to_token = index_to_token or [
            {"index": idx, "table": None, "column": None, "value": tok}
            for tok, idx in self.special_tokens.items()
        ]

        # Register table masks as PyTorch buffers for automatic GPU/CPU transfer
        if table_masks:
            for table_name, mask_tensor in table_masks.items():
                self.register_buffer(f"mask_{table_name}", mask_tensor)

    @property
    def vocab_size(self) -> int:
        return len(self.index_to_token)

    def get_table_mask(self, table_name: str) -> torch.Tensor:
        """Returns the boolean tensor mask for a given table."""
        return getattr(self, f"mask_{table_name}")

    def encode_value(self, table: str, column: str, value: Optional[str]) -> int:
        """Maps a (table, column, value) string triplet to a token ID."""
        if value is None:
            return self.special_tokens["<NULL>"]
        return self.vocab_map.get(
            (table, column, str(value)), self.special_tokens["<UNK>"]
        )

    def encode(self, table: str, column=None, value=None):
        """Encode a table, column, value, or a batch of values."""
        if column is None:
            return self.vocab_map.get((table, None, None), self.special_tokens["<UNK>"])
        if value is None:
            return self.vocab_map.get((table, column, None), self.special_tokens["<UNK>"])
        if isinstance(value, (list, tuple)):
            return torch.tensor(
                [self.encode_value(table, column, item) for item in value],
                dtype=torch.long,
            )
        return self.encode_value(table, column, value)

    def decode(self, token_ids: Union[torch.Tensor, List[int]]) -> List[dict]:
        """Decodes token IDs back to their metadata (table, column, value)."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [
            self.index_to_token[tid]
            if 0 <= tid < len(self.index_to_token)
            else {"index": tid, "table": None, "column": None, "value": "<UNK>"}
            for tid in token_ids
        ]

    def save(self, path: Path) -> None:
        """Saves the PyTorch module state and vocabulary metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "special_tokens": self.special_tokens,
                "vocab_map": self.vocab_map,
                "index_to_token": self.index_to_token,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> HealthcareVocab:
        """Loads a saved vocabulary checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        vocab = cls(
            special_tokens=checkpoint["special_tokens"],
            vocab_map=checkpoint["vocab_map"],
            index_to_token=checkpoint["index_to_token"],
        )

        # Pre-register mask buffers stored in state_dict so PyTorch expects them
        state_dict = checkpoint["state_dict"]
        for key, tensor in state_dict.items():
            if key.startswith("mask_") and not hasattr(vocab, key):
                vocab.register_buffer(key, torch.zeros_like(tensor))

        vocab.load_state_dict(state_dict)
        return vocab


def build_pytorch_vocab(
    database_path: Path = DATABASE_PATH,
) -> HealthcareVocab:
    """Builds the vocabulary from DuckDB and converts masks to PyTorch Tensors."""
    special_tokens = {"<PAD>": 0, "<UNK>": 1, "<NULL>": 2, "<NaT>": 3}
    next_index = len(special_tokens)

    vocab_map: Dict[tuple[str, str, str], int] = {}
    index_to_token: List[dict] = [
        {"index": idx, "table": None, "column": None, "value": tok}
        for tok, idx in special_tokens.items()
    ]
    table_token_indices: Dict[str, List[int]] = {}

    with duckdb.connect(str(database_path), read_only=True) as con:
        available_tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }

        for table_name, columns in VOCAB_COLUMNS.items():
            if table_name not in available_tables:
                continue

            table_token_indices[table_name] = []
            available_cols = {
                row[0]
                for row in con.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ?",
                    [table_name],
                ).fetchall()
            }

            for output_column, source_column in columns.items():
                if output_column == "event_type":
                    values = [source_column]
                elif source_column not in available_cols:
                    continue
                else:
                    query = f'SELECT DISTINCT CAST("{source_column}" AS VARCHAR) FROM "{table_name}" WHERE "{source_column}" IS NOT NULL ORDER BY 1'
                    values = [row[0] for row in con.execute(query).fetchall()]

                for val in values:
                    token_idx = next_index
                    next_index += 1

                    vocab_map[(table_name, output_column, str(val))] = token_idx
                    index_to_token.append(
                        {
                            "index": token_idx,
                            "table": table_name,
                            "column": output_column,
                            "value": val,
                        }
                    )
                    table_token_indices[table_name].append(token_idx)

    # Convert table masks to boolean PyTorch Tensors
    table_masks: Dict[str, torch.Tensor] = {}
    for table_name, indices in table_token_indices.items():
        mask = torch.zeros(next_index, dtype=torch.bool)
        if indices:
            mask[torch.tensor(indices, dtype=torch.long)] = True
        table_masks[table_name] = mask

    return HealthcareVocab(
        special_tokens=special_tokens,
        vocab_map=vocab_map,
        index_to_token=index_to_token,
        table_masks=table_masks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    vocab = build_pytorch_vocab(args.database)
    vocab.save(args.output)
    print(f"Wrote {vocab.vocab_size:,} tokens to {args.output}")


if __name__ == "__main__":
    main()