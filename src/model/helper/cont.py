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
            freq_bands = (2.0 ** torch.arange(-num_frequencies//2, num_frequencies//2, dtype=torch.float32)) * torch.pi
        frequency_count = len(freq_bands)
        self.register_buffer("frequencies", freq_bands)

        self.projection = nn.Sequential(
            nn.Linear(input_dim * (1 + 2 * frequency_count), output_dim),
            nn.GELU(),
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
                # There is no exact Kaiming gain for GELU.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
