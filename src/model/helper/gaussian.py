import torch
import torch.nn as nn

class GaussianMixtureEstimator(nn.Module):
	def __init__(self, input_dim, n_gaussians = 10):
		super(GaussianMixtureEstimator, self).__init__()
		self.n_gaussians = n_gaussians
		self.input_dim = input_dim

		self.estimator = nn.Linear(input_dim, n_gaussians * 3)  # Each Gaussian has mean, log std, and weight
		self.reset_parameters()
	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.estimator.weight, nonlinearity='linear')
		if self.estimator.bias is not None:
			nn.init.zeros_(self.estimator.bias)

	def forward(self, x):
		# x shape: (batch_size, input_dim)
		params = self.estimator(x)  # Shape: (batch_size, n_gaussians * 3)
		params = params.view(*params.shape[:-1], self.n_gaussians, 3)

		means = params[..., 0]  # Shape: (..., n_gaussians)
		log_stds = params[..., 1]  # Shape: (..., n_gaussians)
		log_weights = params[..., 2]  # Shape: (..., n_gaussians)

		# Store normalized log-weights so likelihood evaluation uses the same
		# parameterization as the model output without taking another log.
		log_weights = torch.log_softmax(log_weights, dim=-1)

		return torch.cat([means, log_stds, log_weights], dim=-1)  # (..., 3 * n_gaussians)

	@staticmethod
	def point_prediction(distribution, n_gaussians):
		"""Return the deterministic point prediction (the mixture mean)."""
		means = distribution[..., :n_gaussians]
		weights = torch.softmax(distribution[..., 2 * n_gaussians:], dim=-1)
		return (weights * means).sum(dim=-1)

	@staticmethod
	def entropy(distribution, n_gaussians):
		"""Approximate mixture entropy using the weighted component entropies.

		The exact entropy of a Gaussian mixture has no closed form. This
		quantity is stable, cheap to log, and is a useful uncertainty proxy.
		"""
		log_stds = distribution[..., n_gaussians:2 * n_gaussians].clamp(-7.0, 5.0)
		weights = torch.softmax(distribution[..., 2 * n_gaussians:], dim=-1)
		component_entropy = log_stds + 0.5 * distribution.new_tensor(2.0 * torch.pi * torch.e).log()
		return (weights * component_entropy).sum(dim=-1)

	@staticmethod
	def log_likelihood(x, distribution, n_gaussians):
		# This loss is called under AMP. In fp16, a valid small standard
		# deviation can make exp(-2 * log_std) and the squared residual
		# overflow before logsumexp gets a chance to stabilize it. Keep the
		# complete likelihood calculation in fp32.
		x = x.float()
		distribution = distribution.float()
		means = distribution[..., :n_gaussians]
		log_stds = distribution[..., n_gaussians:2*n_gaussians]
		log_weights = distribution[..., 2*n_gaussians:]
		log_stds = log_stds.clamp(min=-7.0, max=5.0)

		x = x.reshape(*x.shape, *([1] * (distribution.ndim - x.ndim)))

		# log N(x | mean, std) evaluated without materializing variances.
		standardized = ((x - means) * torch.exp(-log_stds)).clamp(-1e4, 1e4)
		log_probs = -log_stds - 0.5 * standardized.square()
		log_probs = log_probs - 0.5 * log_stds.new_tensor(2.0 * torch.pi).log()
		log_probs = log_probs + log_weights

		return torch.logsumexp(log_probs, dim=-1)  # Sum over Gaussians

		

import torch
import torch.distributions as D

class MixtureDensityNetworkHelpers:
    
    @staticmethod
    def get_distribution(distribution_params, n_gaussians):
        """Build PyTorch MixtureSameFamily object from raw network outputs."""
        means = distribution_params[..., :n_gaussians]
        # Assuming index [n_gaussians : 2*n_gaussians] are log_stds
        stds = torch.exp(distribution_params[..., n_gaussians : 2 * n_gaussians])
        logits = distribution_params[..., 2 * n_gaussians:]
        
        mix = D.Categorical(logits=logits)
        comp = D.Normal(loc=means, scale=stds)
        return D.MixtureSameFamily(mix, comp)

    @staticmethod
    def loss_function(distribution_params, n_gaussians, target):
        """1. WHAT IT OPTIMIZES: Negative Log-Likelihood"""
        gmm = MixtureDensityNetworkHelpers.get_distribution(distribution_params, n_gaussians)
        return -gmm.log_prob(target).mean()

    @staticmethod
    def point_prediction(distribution_params, n_gaussians):
        """2. EXPECTATION: Deterministic mixture mean E[Y|X]"""
        means = distribution_params[..., :n_gaussians]
        weights = torch.softmax(distribution_params[..., 2 * n_gaussians:], dim=-1)
        return (weights * means).sum(dim=-1)

    @staticmethod
    def sample(distribution_params, n_gaussians, sample_shape=torch.Size()):
        """3. SAMPLING: Draw random points from learned density p(y|x)"""
        gmm = MixtureDensityNetworkHelpers.get_distribution(distribution_params, n_gaussians)
        return gmm.sample(sample_shape)