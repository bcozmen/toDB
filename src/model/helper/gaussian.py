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
	def log_likelihood(x, distribution, n_gaussians):
		means = distribution[..., :n_gaussians]
		log_stds = distribution[..., n_gaussians:2*n_gaussians]
		log_weights = distribution[..., 2*n_gaussians:]
		log_stds = log_stds.clamp(min=-7.0, max=5.0)

		x = x.reshape(*x.shape, *([1] * (distribution.ndim - x.ndim)))

		# log N(x | mean, std) evaluated without materializing variances.
		log_probs = -log_stds - 0.5 * (x - means).square() * torch.exp(-2.0 * log_stds)
		log_probs = log_probs - 0.5 * log_stds.new_tensor(2.0 * torch.pi).log()
		log_probs = log_probs + log_weights

		return torch.logsumexp(log_probs, dim=-1)  # Sum over Gaussians

		