import torch
import torch.nn as nn

PERIODS_IN_YEARS = torch.tensor([
    100.0,           # Century
    10.0,            # Decade
    1.0,             # Year
    0.25,            # 3 Months
    1.0 / 12.0,      # Month
    7.0 / 365.0,     # Week
    1.0 / 365.0,     # Day
    1.0 / 365.0 / 24.0, # Hour
    1.0 / 365.0 / 24.0 / 60.0, # Minute
    1.0 / 365.0 / 24.0 / 60.0 / 60.0, # Second
], dtype=torch.float32)


class NumericProjectionWithFrequency(nn.Module):
    def __init__(self, input_dim, output_dim, num_frequencies=10, time=False):
        super(NumericProjectionWithFrequency, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies
        self.time = time

        if time:
            self.num_frequencies = len(PERIODS_IN_YEARS)
            freq_bands = (100 * torch.pi) / PERIODS_IN_YEARS
        else:
            freq_bands = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * torch.pi
        self.register_buffer("frequencies", freq_bands)

        self.projection = nn.Sequential(
            nn.Linear(input_dim * (1 + 2 * num_frequencies), output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.reset_parameters()

    def forward(self, x):
        # x shape: (batch_size, num_non_categorical, input_dim)
        x_expanded = x.unsqueeze(-1) * self.frequencies  # Shape: (batch_size, num_non_categorical, input_dim, num_frequencies)
        x_sin = torch.sin(x_expanded).flatten(start_dim=-2)  # Shape: (batch_size, num_non_categorical, input_dim * num_frequencies)
        x_cos = torch.cos(x_expanded).flatten(start_dim=-2)  # Shape: (batch_size, num_non_categorical, input_dim * num_frequencies)
        x_proj = torch.cat([x, x_sin, x_cos], dim=-1)        # Shape: (batch_size, num_non_categorical, input_dim * (1 + 2 * num_frequencies))
        return self.projection(x_proj)  
    def reset_parameters(self):
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class QuestionEmbedding(nn.Module):
    def __init__(self, num_questions, embedding_dim):
        super(QuestionEmbedding, self).__init__()
        self.questions = nn.Parameter(torch.randn(num_questions, embedding_dim) * 0.02)

class Embedding(nn.Module):
    def __init__(self, embedding_dim, dictionary, num_frequencies=10):
        super(Embedding, self).__init__()
        self.dictionary = dictionary
        self.embedding_dim = embedding_dim

        self.categorical = nn.Embedding(num_embeddings=len(dictionary), embedding_dim=embedding_dim)
        self.numeric = NumericProjectionWithFrequency(input_dim=1, output_dim=embedding_dim, num_frequencies=num_frequencies, time=False)
        self.time = NumericProjectionWithFrequency(input_dim=1, output_dim=embedding_dim, num_frequencies=num_frequencies, time=True)
    
        #self.cls_token = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, X):
        # 1. Column 0: First categorical feature -> (..., Seq, 1, embedding_dim)
        cat_first = self.get_categorical_embeddings(X[..., [0]])
        
        # 2. Columns 1 & 2: Numeric feature & presence mask -> (..., Seq, 1, embedding_dim)
        numeric = self.get_numeric_embeddings(X[..., [1]], X[..., [2]])
        
        # 3. Columns 3, 4, 5: Start time, duration, dT -> (..., Seq, 3, embedding_dim)
        time = self.get_time_embeddings(X[..., 3:6])
        
        # 4. Columns 6 & 8: Table ID & Event ID -> (..., Seq, 2, embedding_dim)
        cat_last = self.get_categorical_embeddings(X[..., [6, 8]])
        
        # Concatenate features sequentially along dim=-2 -> (..., Seq, 7, embedding_dim)
        embeddings = torch.cat([cat_first, numeric, time, cat_last], dim=-2)

        # Prepend/Append CLS Token along sequence dim=-3 -> (..., Seq + 1, 7, embedding_dim)
        cls_token = torch.zeros(*embeddings.shape[:-3], 1, embeddings.shape[-2], self.embedding_dim, device=embeddings.device)
        cls_token[..., 0, 0, :] = self.cls_token
        embeddings = torch.cat([cls_token, embeddings], dim=-3)
        
        return embeddings

    def get_categorical_embeddings(self, x_cat):
        return self.categorical(x_cat.long())

    def get_numeric_embeddings(self, val, mask):
        num_embeddings = torch.zeros(*val.shape[:-1], 1, self.embedding_dim, device=val.device, dtype=torch.float32)
        
        val_flat = val.squeeze(-1)
        mask_flat = mask.bool().squeeze(-1)
        
        nan_mask = torch.isnan(val_flat)
        valid_mask = mask_flat & ~nan_mask
        nan_num_mask = mask_flat & nan_mask

        if valid_mask.any():
            valid_vals = val[valid_mask]  # shape: (N, 1)
            num_embeddings[valid_mask, 0] = self.numeric(valid_vals)

        if nan_num_mask.any():
            none_idx = torch.tensor(self.dictionary.encode(None), device=val.device, dtype=torch.long)
            num_embeddings[nan_num_mask, 0] = self.categorical(none_idx)
            
        return num_embeddings

    def get_time_embeddings(self, time_vals):
        nan_time_mask = torch.isnan(time_vals) | torch.isinf(time_vals)
        time_vals_clean = torch.where(nan_time_mask, torch.zeros_like(time_vals), time_vals)
        
        time_embeddings = self.time(time_vals_clean.unsqueeze(-1))  # shape: (..., 3, embedding_dim)
        
        nan_embedding = self.categorical(
            torch.tensor(self.dictionary.encode(float('inf')), device=time_vals.device, dtype=torch.long)
        )
        return torch.where(nan_time_mask.unsqueeze(-1), nan_embedding, time_embeddings)

