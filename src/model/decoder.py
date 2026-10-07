import torch
import torch.nn as nn

from .future import FactorizedEventHead, FutureEncounterPredictor


class PredictionDecoder(nn.Module):
    """Decode one causal representation into all prediction tasks.

    ``FactorizedEventHead`` replaces the former collection of question slots.
    Conditioning is explicit and teacher forcing is supplied through ``X``.
    """

    def __init__(self, embedding_dim, condition_dim, dictionary_size,
                 num_gaussians=10):
        super().__init__()
        self.factorized_head = FactorizedEventHead(
            embedding_dim, dictionary_size, condition_dim, num_gaussians
        )
        #self.future_predictor = FutureEncounterPredictor(
        #    embedding_dim, num_gaussians, dictionary_size
        #)
        self.future_predictor = FactorizedEventHead(
            embedding_dim, dictionary_size, condition_dim, num_gaussians, table_active=False
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, X):
        latent, x = X

        next_hidden = latent[..., :-1, :]
        class_hidden = latent[..., 1:, :]
        class_logits = self.classifier(class_hidden)
        next_event = self.factorized_head((next_hidden, x))
        time, _, code = self.future_predictor((next_hidden, x))
        return class_logits, next_event, (time, code)

    @torch.no_grad()
    def generate(self, hidden):
        """Generate one next event from a causal hidden representation."""
        return self.factorized_head.generate(hidden)



