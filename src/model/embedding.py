import torch
import torch.nn as nn

from .helper import NumericProjectionWithFrequency

class Embedding(nn.Module):
    def __init__(self, embedding_dim, dictionary, embedding_length=6, num_frequencies=10, dropout=0.1):
        super(Embedding, self).__init__()
        self.dictionary = dictionary
        self.embedding_dim = embedding_dim
        self.embedding_length = embedding_length  # Number of features to embed (categorical + numeric + time)
        self.latent_dim = self.embedding_dim // self.embedding_length

        self.categorical = nn.Embedding(num_embeddings=len(dictionary), embedding_dim=self.latent_dim)
        self.numeric = NumericProjectionWithFrequency(input_dim=1, output_dim=self.latent_dim, num_frequencies=num_frequencies, time=False)
        self.time = NumericProjectionWithFrequency(input_dim=1, output_dim=self.latent_dim, num_frequencies=num_frequencies, time=True)
        self.source_target = nn.Embedding(num_embeddings=2, embedding_dim=self.latent_dim)  # For source/target indicator

        self.time_columns = [3]  # Start time

        self.projector = nn.Linear(self.latent_dim * self.embedding_length, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.dropout = nn.Dropout(dropout)
        
        self.empty_token = nn.Parameter(torch.randn(1,1,1,self.latent_dim) * 0.02)
        
        self.reset_parameters()



    def reset_parameters(self):
        nn.init.normal_(self.categorical.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.source_target.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.projector.weight, nonlinearity='linear')
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)
    def forward(self, X):
        #x shape = (batch_size, embedding_length, seq_length)
        X, m = X

        cat_first = self.get_categorical_embeddings(X[..., [0],:])
        numeric = self.get_numeric_embeddings(X[..., [1], :], X[..., [2], :])
        time = self.get_time_embeddings(X[..., self.time_columns, :])
        cat_last = self.get_categorical_embeddings(X[..., [6, 7], :])
        # Row 9 marks sampled target positions. Both classes receive the same
        # sampled marker pattern; only negative samples replace the marked tokens.
        source_target = self.source_target(X[..., [9], :].long())
        embeddings = torch.cat([cat_first, numeric, time, cat_last, source_target], dim=1)
        #embeddings shape = (batch_size, embedding_length, seq_length, embedding_dim)

        #pad embeddings with the empty token
        empty_token_expanded = self.empty_token.expand(*embeddings.shape[:2], 1, -1)
        embeddings = torch.cat([empty_token_expanded, embeddings], dim=-2)

        #embeddings shape = (batch_size, embedding_length, seq_length, embedding_dim)
        tokens = self.projector(embeddings.permute(0, 2, 1, 3).flatten(start_dim=-2))  # Project to embedding_dim
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)
        

        #Pad embeddings with the empty token

        #m shape batch, seq
        m = torch.cat([torch.zeros((m.shape[0], 1), device=m.device, dtype=m.dtype), m], dim=-1)
        return tokens, embeddings, m


    def get_categorical_embeddings(self, x_cat):
        return self.categorical(x_cat.long())

    def get_numeric_embeddings(self, val, mask):
        numeric_mask = torch.isfinite(val) & mask.bool()

        # Project every position using finite placeholder values, then select the
        # result with torch.where. This avoids dynamic nonzero/index_put graphs
        # under torch.compile and keeps the backward pass free of in-place writes.
        safe_values = torch.asinh(torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
        numeric_embeddings = self.numeric(safe_values.unsqueeze(-1))
        missing_embedding = self.categorical(
            torch.tensor(self.dictionary.encode(None), device=val.device, dtype=torch.long)
        ).to(numeric_embeddings.dtype)
        return torch.where(numeric_mask.unsqueeze(-1), numeric_embeddings, missing_embedding)

    def get_numeric_embeddings_old(self, val, mask):
        #val.shape = (batch_size, 1, seq_lenth)
        val = val.clone()
        nan_mask = torch.isnan(val) | torch.isinf(val)
        val[nan_mask] = 0.0  # Replace NaN and Inf with 0 for embedding
        embeddings = self.numeric(val.unsqueeze(-1))  # shape: (batch_size, 1, seq_length, embedding_dim)
        missing_embedding = self.categorical(torch.tensor(self.dictionary.encode(None), device=val.device, dtype=torch.long))
        embeddings = torch.where(
            mask.bool().unsqueeze(-1), embeddings, torch.zeros_like(embeddings)
        )
        embeddings = torch.where(
            nan_mask.unsqueeze(-1), missing_embedding.to(embeddings.dtype), embeddings
        )
        return embeddings
    
    def get_time_embeddings(self, time_vals):
        nan_mask = ~torch.isfinite(time_vals)
        safe_time_vals = torch.nan_to_num(time_vals, nan=0.0, posinf=0.0, neginf=0.0)
        embeddings = self.time(safe_time_vals.unsqueeze(-1))  # shape: (batch_size, 1, seq_length, embedding_dim)
        missing_embedding = self.categorical(torch.tensor(self.dictionary.encode(float('inf')), device=time_vals.device, dtype=torch.long))
        embeddings = torch.where(
            nan_mask.unsqueeze(-1), missing_embedding.to(embeddings.dtype), embeddings
        )
        return embeddings


