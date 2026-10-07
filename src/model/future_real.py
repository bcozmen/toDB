import torch
import torch.nn as nn

from .helper import GaussianMixtureEstimator

IMPORTANT_ENCOUNTER_INDICES = [984, 1034, 1016, 1036, 1037, 1038, 1043, 1044, 990, 991, 992, 995, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1014, 1021, 1024, 1033, 1041, 1042, 986]


def future_time_target(x):
    """Return real time to the closest important encounter per position."""
    observed = x[..., 8, :] > 0
    ages = torch.nan_to_num(x[..., 3, :], nan=0.0, posinf=0.0, neginf=0.0)
    codes = x[..., 0, :].long()
    important = torch.isin(
        codes, x.new_tensor(IMPORTANT_ENCOUNTER_INDICES).long()
    ) & observed

    positions = torch.arange(x.size(-1), device=x.device).expand_as(codes)
    sentinel = x.size(-1)
    target_positions = torch.flip(
        torch.cummin(
            torch.flip(torch.where(important, positions, sentinel), (1,)), 1
        ).values,
        (1,),
    )
    has_target = target_positions.lt(sentinel)
    target_positions = target_positions.clamp_max(x.size(-1) - 1)
    prediction_positions = (positions - 1).clamp_min(0)
    prediction_ages = ages.gather(1, prediction_positions)
    target_ages = ages.gather(1, target_positions)
    target_times = (target_ages - prediction_ages).clamp_min(0.0)
    return torch.where(has_target, target_times, torch.zeros_like(target_times))

class FactorizedEventHead(nn.Module):
    """Autoregressive head for the next event's fields.

    The head follows
    p(time, table, code | h) = p(time | h)
    p(table | h, time) p(code | h, time, table)
    Value prediction is intentionally deferred.
    """

    def __init__(self, embedding_dim, dictionary_size, condition_dim,
                 num_gaussians=10, table_active=True):
        super().__init__()
        self.dictionary_size = dictionary_size
        self.num_gaussians = num_gaussians
        
        
        self.time_conditioner = nn.Sequential(
            nn.Linear(1, condition_dim), 
            nn.GELU(), 
            nn.Linear(condition_dim, condition_dim)
        )
        
        self.table_condition_embedding = nn.Embedding(dictionary_size, condition_dim)
        
        self.time_head = GaussianMixtureEstimator(embedding_dim, num_gaussians)
        self.table_active = table_active
        if self.table_active:
            self.table_head = nn.Linear(embedding_dim + condition_dim, dictionary_size)

        num_condition = 2 * condition_dim if self.table_active else condition_dim
        self.code_head = nn.Linear(embedding_dim + num_condition, dictionary_size)
        


    def forward(self, X):
        hidden, x = X
        table_target = x[..., 6, :]
        time_params = self.time_head(hidden)
        # Channel 5 is the next-event target, not the future-encounter target.
        time_for_condition = future_time_target(x)
        time_context = self.time_conditioner(time_for_condition.unsqueeze(-1))

        if self.table_active:
            table_logits = self.table_head(torch.cat((hidden, time_context), dim=-1))
            table_context = self.table_condition_embedding(table_target.long())
            code_logits = self.code_head(torch.cat((hidden, time_context, table_context), dim=-1))
            return time_params, table_logits, code_logits
        else:
            code_logits = self.code_head(torch.cat((hidden, time_context), dim=-1))
            return time_params, None, code_logits

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
    """Predict continuous delay and code distributions for future encounters."""

    def __init__(self, embedding_dim, n_gaussians, dictionary_size):
        super().__init__()
        self.future = GaussianMixtureEstimator(embedding_dim, n_gaussians=n_gaussians)
        self.code = nn.Linear(embedding_dim, dictionary_size)
        self.dictionary_size = dictionary_size

    def forward(self, hidden):
        future_params = self.future(hidden)
        code_logits = self.code(hidden)
        return future_params, code_logits

