import torch
import torch.nn as nn

from .helper import GaussianMixtureEstimator


class FactorizedEventHead(nn.Module):
    """Autoregressive head for the next event's fields.

    The head follows
    p(time, table, code | h) = p(time | h)
    p(table | h, time) p(code | h, time, table)
    Value prediction is intentionally deferred.
    """

    def __init__(self, embedding_dim, dictionary_size, condition_dim, num_gaussians=10):
        super().__init__()
        self.dictionary_size = dictionary_size
        
        
        self.time_conditioner = nn.Sequential(nn.Linear(1, condition_dim), nn.GELU(), nn.Linear(condition_dim, condition_dim))
        
        self.table_condition_embedding = nn.Embedding(dictionary_size, condition_dim)
        
        self.time_head = GaussianMixtureEstimator(embedding_dim, num_gaussians)
        self.table_head = nn.Linear(embedding_dim + condition_dim, dictionary_size)
        self.code_head = nn.Linear(embedding_dim + 2 * condition_dim, dictionary_size)

    def forward(self, X):
        hidden, x = X
        time_target, table_target, code_target = x[..., 3, :], x[..., 6, :], x[..., 0, :]
        time_params = self.time_head(hidden)
        time_context = self.time_conditioner(time_target.unsqueeze(-1))

        table_logits = self.table_head(torch.cat((hidden, time_context), dim=-1))
        table_context = self.table_condition_embedding(table_target.long())

        code_logits = self.code_head(torch.cat((hidden, time_context, table_context), dim=-1))
        return time_params, table_logits, code_logits

    @staticmethod
    def mixture_mean(params, num_gaussians):
        """Return the mean of the predicted Gaussian mixture."""
        means = params[..., :num_gaussians]
        log_weights = params[..., 2 * num_gaussians:]
        weights = torch.softmax(log_weights, dim=-1)
        return (weights * means).sum(dim=-1)

    def generate(self, hidden):
        """Generate one event without requiring ground-truth target fields.

        This deterministic path uses the mixture mean and argmax categorical
        predictions. Sampling can be added by replacing these selections.
        """
        time_params = self.time_head(hidden)
        time = self.mixture_mean(time_params, self.time_head.n_gaussians)
        time_context = self.time_conditioner(time.unsqueeze(-1))

        table_logits = self.table_head(torch.cat((hidden, time_context), dim=-1))
        table = table_logits.argmax(dim=-1)
        table_context = self.table_condition_embedding(table)

        code_logits = self.code_head(
            torch.cat((hidden, time_context, table_context), dim=-1)
        )
        code = code_logits.argmax(dim=-1)
        return {
            "time": time,
            "table": table,
            "code": code,
            "time_params": time_params,
            "table_logits": table_logits,
            "code_logits": code_logits,
        }


class FutureEncounterPredictor(nn.Module):
    """Predict discrete-time hazards for the next encounter.

    Each output corresponds to the interval ending at one of the configured
    horizon boundaries. The hazard in interval ``k`` is conditional on no
    encounter having occurred in an earlier interval.
    """

    def __init__(self, embedding_dim, num_horizons, dictionary_size):
        super().__init__()
        self.num_horizons = num_horizons
        self.hazard = nn.Linear(embedding_dim, num_horizons)
        self.code = nn.Linear(embedding_dim, num_horizons * dictionary_size)
        self.dictionary_size = dictionary_size

    def forward(self, hidden):
        hazard_logits = self.hazard(hidden)
        code_logits = self.code(hidden).view(*hidden.shape[:-1], self.num_horizons, self.dictionary_size)
        return hazard_logits, code_logits

    @staticmethod
    def cumulative_risk(hazard_logits):
        """Convert interval hazards into cumulative encounter probabilities."""
        hazards = torch.sigmoid(hazard_logits)
        survival = torch.cumprod(1.0 - hazards, dim=-1)
        return 1.0 - survival
