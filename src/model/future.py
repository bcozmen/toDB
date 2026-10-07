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

    def __init__(self, embedding_dim, dictionary_size, condition_dim,
                 num_gaussians=10, table_active=True,
                 deterministic=False):
        super().__init__()
        self.dictionary_size = dictionary_size
        self.num_gaussians = num_gaussians
        self.deterministic = deterministic
        self.table_active = table_active
        
        self.time_head = GaussianMixtureEstimator(embedding_dim, num_gaussians)
        
        self.time_conditioner = nn.Sequential(
            nn.Linear(1, condition_dim), 
            nn.GELU(), 
            nn.Linear(condition_dim, condition_dim),
            nn.LayerNorm(condition_dim),
        )
        
        self.table_condition_embedding = nn.Sequential(
            nn.Embedding(dictionary_size, condition_dim),
            nn.LayerNorm(condition_dim),
        )
        
        if self.table_active:
            self.table_head = nn.Sequential(
                nn.LayerNorm(embedding_dim + condition_dim),
                nn.Linear(embedding_dim + condition_dim, dictionary_size),
            )

        num_condition = 2 * condition_dim if self.table_active else condition_dim
        self.code_head = nn.Sequential(
            nn.LayerNorm(embedding_dim + num_condition),
            nn.Linear(embedding_dim + num_condition, dictionary_size),
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize table embedding table within its sequential container
        nn.init.normal_(self.table_condition_embedding[0].weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X):
        hidden, x = X
        time_params = self.time_head(hidden)
        # Detach time_for_condition so downstream table/code losses do not distort the GMM density estimator.
        # Factorized decomposition: p(time, table, code | h) = p(time | h) p(table | h, time) p(code | h, time, table)
        time_for_condition = self.predict_time(time_params).detach()
        time_context = self.time_conditioner(time_for_condition.unsqueeze(-1))

        if self.table_active:
            table_logits = self.table_head(torch.cat((hidden, time_context), dim=-1))
            # Softmax expectation over table embeddings in fp32 to prevent AMP underflow/overflow
            table_probs = torch.softmax(table_logits.float(), dim=-1).detach()
            table_context = table_probs @ self.table_condition_embedding[0].weight.float()
            table_context = self.table_condition_embedding[1](table_context.to(dtype=hidden.dtype))

            code_logits = self.code_head(torch.cat((hidden, time_context, table_context), dim=-1))
            return time_params, table_logits, code_logits
        else:
            code_logits = self.code_head(torch.cat((hidden, time_context), dim=-1))
            return time_params, None, code_logits

    @staticmethod
    def sample_time(params, num_gaussians):
        """Sample a finite time from the predicted Gaussian mixture.

        Sampling is performed in float32 even when the model forward pass uses
        AMP.  Non-finite distribution parameters are replaced before the
        categorical draw, and the standard deviation is bounded to prevent
        rare extreme samples from destabilizing the conditioning path.
        """
        if not torch.isfinite(params).all():
            raise FloatingPointError("Non-finite GMM parameters in sample_time")

        output_dtype = params.dtype
        sampling_params = params.float()

        means = sampling_params[..., :num_gaussians]
        log_stds = sampling_params[..., num_gaussians:2 * num_gaussians].clamp(-7.0, 4.5)
        weights = torch.softmax(sampling_params[..., 2 * num_gaussians:], dim=-1)

        component = torch.multinomial(weights.reshape(-1, num_gaussians), 1)
        component = component.reshape(weights.shape[:-1])
        mean = means.gather(-1, component.unsqueeze(-1)).squeeze(-1)
        log_std = log_stds.gather(-1, component.unsqueeze(-1)).squeeze(-1)
        sample = mean + log_std.exp() * torch.randn_like(mean)
        return sample.to(dtype=output_dtype)

    @staticmethod
    def sample_categorical(logits):
        """Sample one category independently at every prediction position."""
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite categorical logits in sample_categorical")
        logits = logits.float()
        return torch.distributions.Categorical(logits=logits).sample()

    def select_categorical(self, logits):
        if self.deterministic:
            return logits.argmax(dim=-1)
        return self.sample_categorical(logits)

    def predict_time(self, params):
        if self.deterministic or self.training:
            return self.mixture_mean(params, self.num_gaussians)
        return self.sample_time(params, self.num_gaussians)

    @staticmethod
    def mixture_mean(params, num_gaussians):
        """Return the mean of the predicted Gaussian mixture."""
        means = params[..., :num_gaussians]
        log_weights = params[..., 2 * num_gaussians:]
        weights = torch.softmax(log_weights, dim=-1)
        return (weights * means).sum(dim=-1)

    def generate(self, hidden):
        """Generate one event without requiring ground-truth target fields.

        This path samples time from the GMM and samples both categorical
        fields from their predicted distributions.
        """
        time_params = self.time_head(hidden)
        time = self.predict_time(time_params)
        time_context = self.time_conditioner(time.unsqueeze(-1))

        if self.table_active:
            table_logits = self.table_head(torch.cat((hidden, time_context), dim=-1))
            table = self.select_categorical(table_logits)
            table_context = self.table_condition_embedding(table)

            code_logits = self.code_head(
                torch.cat((hidden, time_context, table_context), dim=-1)
            )
            code = self.select_categorical(code_logits)
            return {
                "time": time,
                "table": table,
                "code": code,
                "time_params": time_params,
                "table_logits": table_logits,
                "code_logits": code_logits,
            }
        else:
            code_logits = self.code_head(torch.cat((hidden, time_context), dim=-1))
            code = self.select_categorical(code_logits)
            return {
                "time": time,
                "code": code,
                "time_params": time_params,
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

