import torch
import torch.nn as nn
import copy

from deeppy import BaseModel, Network, Optimizer
from . import CausalMaskedTransformer, Embedding, \
	PredictionDecoder, GaussianMixtureEstimator, StatLogger
from torch.nn.attention import SDPBackend, sdpa_kernel

from .. import HealthCareDictionary


# Time is normalized by ten years in the dataset loader.
DEFAULT_FUTURE_HORIZONS = (
	7 / 3650,       # 1 week
	30 / 3650,      # 1 month
	182 / 3650,     # 6 months
	365 / 3650,     # 1 year
	1825 / 3650,    # 5 years
	3650 / 3650,    # 10 years
)


class DBTransformer(BaseModel):
	dependencies = [Network, Optimizer]
	def __init__(self, dictionary_path, optimizer_params,
				num_frequencies=10, num_embedding=6, num_gaussians=10,
				d_model = 64, nhead= 8, num_encoder_layers = 4, dim_feedforward = 256, 
				 context_size = 2048, dropout = 0.1, activation = "gelu", 
				 last_layer_norm = True, use_checkpointing=False):

		super(DBTransformer, self).__init__()
		self.optimizer_params = copy.deepcopy(optimizer_params)
		self.dictionary = HealthCareDictionary(dictionary_path)  # Add this line to load the dictionary
		
		
		self.num_frequencies = num_frequencies
		self.num_embedding = num_embedding
		self.use_checkpointing = use_checkpointing
		self.num_gaussians = num_gaussians
		self.condition_dim = d_model // num_embedding
		self.future_horizons = tuple(DEFAULT_FUTURE_HORIZONS)
		
		self.context_size = context_size
		self.encoder_layer_args = {"d_model": d_model, "nhead": nhead, "dim_feedforward": dim_feedforward, "dropout": dropout, "activation": activation}
		self.encoder_args = {"num_layers": num_encoder_layers, "norm": nn.LayerNorm(d_model) if last_layer_norm else None}

	def forward(self, X):
		#T (Batch, Seq_len, Features)
		x, m = X
		tokens, e_raw, m = self.embedding_network((x,m))
		
		with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
			latent = self.transformer_network((tokens, m))
		return self.decoder_network((latent, x))
	
	def get_loss(self, X):
		x, m, labels = X
		class_logits, next_event, future_encounter = self.forward((x, m))

		loss = self.classification_task(class_logits, labels, m)
		next_losses = self.next_event_losses(x, m, labels, *next_event) 
		future_losses = self.future_encounter_losses(x, m, *future_encounter)

		total_loss = loss + next_losses.sum() + future_losses.sum()
		losses = torch.cat((loss.unsqueeze(0), next_losses, future_losses))
		self.stat_logger.log_entropy()
		self.stat_logger.log_loss(losses)
		return total_loss

	def next_event_losses(self, x, padding_mask, labels, time_params, table_logits, code_logits):
		"""Teacher-forced loss for time, table, and code prediction."""
		valid = ~(padding_mask.bool() | labels.eq(0))
		valid = valid & (x[..., 8, :] > 0)
		time = x[..., 3, :]
		table = x[..., 6, :].long()
		code = x[..., 0, :].long()

		time_loss = -GaussianMixtureEstimator.log_likelihood(time, time_params.squeeze(-2), self.num_gaussians)
		table_loss = self.cross_entropy_loss(table_logits.transpose(1, 2), table)
		code_loss = self.cross_entropy_loss(code_logits.transpose(1, 2), code)
		
		valid = valid & torch.isfinite(time)		
		self._log_next_event_metrics(table_logits, code_logits, table, code, valid)
		return torch.stack((time_loss[valid].mean(), table_loss[valid].mean(), code_loss[valid].mean())) if valid.any() else torch.zeros(3, device=x.device)


	def future_encounter_losses(self, x, padding_mask, hazard_logits, code_logits):
		with torch.no_grad():
			hazard_target, hazard_mask, code_target = self._future_targets(x, padding_mask)
		
		# Each hazard is conditional on surviving to the beginning of its
		# interval. The mask excludes intervals hidden by censoring or by an
		# earlier encounter.
		hazard_values = self.bce_loss(hazard_logits, hazard_target)
		hazard_loss = hazard_values[hazard_mask].mean() if hazard_mask.any() else hazard_values.sum() * 0.0
		
		code_values = self.cross_entropy_loss(code_logits.reshape(-1, code_logits.size(-1)), code_target.reshape(-1)).reshape_as(hazard_target)
		event_mask = hazard_mask & hazard_target.bool()
		code_loss = code_values[event_mask].mean() if event_mask.any() else code_values.sum() * 0.0
		self.stat_logger.log_classification(
			hazard_logits[hazard_mask].detach(), hazard_target[hazard_mask],
			"Future Hazard", task="binary"
		)
		self.stat_logger.log_classification(
			code_logits[event_mask].detach(), code_target[event_mask],
			"Future Code", task="multiclass"
		)
		return torch.stack((hazard_loss, code_loss))

	def _log_next_event_metrics(self, table_logits, code_logits, table, code, valid):
		self.stat_logger.log_classification(
			table_logits[valid].detach(), table[valid], "Next Table", task="multiclass"
		)
		self.stat_logger.log_classification(
			code_logits[valid].detach(), code[valid], "Next Code", task="multiclass"
		)

	def _future_targets(self, x, padding_mask):
		"""Build discrete-time survival targets at encounter landmarks.

		The final observed time is treated as a censoring boundary. This is
		conservative for truncated contexts and can later be replaced by exact
		patient follow-up metadata from the repository. Each encounter position
		is a landmark, and the target is the first encounter strictly after it.
		Each output horizon is an interval boundary. The hazard target is one
		only in the interval containing the next encounter; earlier intervals
		are observed negatives and later intervals are not at risk. If no
		encounter is observed, intervals are supervised only when follow-up
		reaches their upper boundary.
		"""
		batch, length = padding_mask.shape
		device = x.device
		num_horizons = len(self.future_horizons)
		encounter_table = self.dictionary.encode("encounters")

		# Find the first future encounter for every sequence position with one
		# reverse cumulative minimum instead of looping over patients and positions.
		positions = torch.arange(length, device=device).expand(batch, -1)
		is_encounter = (~padding_mask.bool()) & (x[..., 8, :] == 1)
		candidate_indices = torch.where(is_encounter, positions, torch.full_like(positions, length))
		next_encounter_including = torch.flip(
			torch.cummin(torch.flip(candidate_indices, dims=[1]), dim=1).values,
			dims=[1],
		)
		next_encounter = torch.cat(
			(
				next_encounter_including[:, 1:],
				torch.full((batch, 1), length, dtype=torch.long, device=device),
			),
			dim=1,
		)
		has_encounter = next_encounter < length
		safe_encounter = next_encounter.clamp_max(length - 1)
		future_time = torch.gather(x[..., 3, :], 1, safe_encounter)
		future_code = torch.gather(x[..., 0, :], 1, safe_encounter).long()

		start = x[..., 3, :]
		valid_landmark = is_encounter & torch.isfinite(start)
		delta = future_time - start

		last_indices = (~padding_mask.bool()).sum(dim=1).clamp_min(1) - 1
		last_time = torch.gather(
			x[..., 3, :], 1, last_indices.unsqueeze(1)
		)
		followup = last_time - start
		horizons = x.new_tensor(self.future_horizons)
		upper = horizons.view(1, 1, num_horizons)
		lower = torch.cat((x.new_zeros(1), horizons[:-1])).view(1, 1, num_horizons)
		delta = delta.unsqueeze(-1)
		has_encounter = has_encounter.unsqueeze(-1)
		finite_delta = torch.isfinite(delta)
		finite_followup = torch.isfinite(followup).unsqueeze(-1)

		# Intervals are (lower, upper], with the first interval being
		# (0, horizons[0]]. This makes each event belong to exactly one bin.
		event_in_interval = (
			has_encounter
			& finite_delta
			& (delta > lower)
			& (delta <= upper)
		)
		observed_to_interval_end = finite_followup & (followup.unsqueeze(-1) >= upper)

		# An interval is at risk when follow-up reaches its start and no earlier
		# event has already occurred. An event in the current interval itself is
		# observed and therefore contributes a positive hazard target.
		at_risk = (
			valid_landmark.unsqueeze(-1)
			& finite_followup
			& (followup.unsqueeze(-1) >= lower)
			& (~has_encounter | (delta > lower))
			& (observed_to_interval_end | event_in_interval)
		)

		hazard = event_in_interval.to(dtype=x.dtype)
		mask = at_risk
		codes = torch.where(
			event_in_interval,
			future_code.unsqueeze(-1).expand(-1, -1, num_horizons),
			torch.zeros(batch, length, num_horizons, dtype=torch.long, device=device),
		)
		return hazard, mask, codes

	#logits shape = (batch_size, context length, 1, 1)
	#y shape = (batch_size, context length)
	def classification_task(self, class_logits, y, padding_mask):
		padding_mask = padding_mask.clone()
		padding_mask[:, 0] = 1  # No classification loss for the first token
		class_logits = class_logits.squeeze(-1).squeeze(-1)  # Shape: (batch_size, context length)
		loss = self.bce_loss(class_logits, y)
		self.stat_logger.log_classification(class_logits[~padding_mask.bool()].detach(), y[~padding_mask.bool()].detach(), "Classification", task="binary")
		
		return loss[~padding_mask.bool()].mean()  

	def back_propagate(self, loss):
		return self.optimizer.step(loss)

	
	# =====================================================================
	#INITIALIZATION FUNCTIONS

	def _init_networks(self):
		self.embedding_network, self.embedding_network_params = self._build_embedding()
		self.transformer_network, self.transformer_network_params = self._build_transformer()
		self.decoder_network, self.decoder_network_params = self._build_decoder()

		self.nets = [self.embedding_network, self.transformer_network, self.decoder_network]
		self.params = [self.embedding_network_params, self.transformer_network_params, self.decoder_network_params]

	def _init_optimizers(self):
		self.optimizer = self._configure_optimizer()
		self.optimizers = [self.optimizer]

	def _init_loss_functions(self):
		self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
		self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
		
		self.loss_functions = [self.bce_loss, self.cross_entropy_loss]

	def _load_loss_functions(self, loss_function):
		self.bce_loss = loss_function[0]
		self.cross_entropy_loss = loss_function[1]

	
	def _init_logger(self):
		tags = [
			"Loss", "Classification", "Future Hazard",
			"Next Table Top-K Accuracy", "Next Table Top-K AUC",
			"Next Code Top-K Accuracy", "Next Code Top-K AUC",
			"Future Code Top-K Accuracy", "Future Code Top-K AUC", "Entropy",
		]
		keys = [
			["Classification Loss", "Next Time Loss", "Next Table Loss", "Next Code Loss",
			 "Future Hazard Loss", "Future Code Loss"],
			["AUC-ROC", "Precision", "Recall", "Accuracy"],
			["AUC-ROC", "Precision", "Recall", "Accuracy"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Classification", "Next Table", "Next Code", "Future Hazard", "Future Code"],
		]
		self.logger = self.create_logger(tags, keys)
		self.stat_logger = StatLogger(self.logger, num_classes=len(self.dictionary), device=self.device)
	

	# =====================================================================
	def _build_decoder(self):
		arch_params = {
			"blocks" : [PredictionDecoder],
			"block_args" : [{
				"embedding_dim": self.encoder_layer_args["d_model"],
				"condition_dim": self.condition_dim,
				"dictionary_size": len(self.dictionary),
				"num_gaussians": self.num_gaussians,
				"num_future_horizons": len(self.future_horizons)
			}]
		}
		network_params = {
			"arch_params" : arch_params,
		}

		return Network(**network_params).to(self.device), network_params
	def _build_transformer(self):
		arch_params = {
			"blocks" : [CausalMaskedTransformer],
			"block_args" : [{
				"encoder_layer_args": self.encoder_layer_args,
				"encoder_args": self.encoder_args,
				"context_size": self.context_size,
				"use_checkpointing": self.use_checkpointing
			}]
		}
		network_params = {
			"arch_params" : arch_params,
		}

		return Network(**network_params).to(self.device), network_params

	def _build_embedding(self):
		arch_params = {
			"blocks" : [Embedding],
			"block_args" : [{
				"embedding_dim": self.encoder_layer_args["d_model"],
				"dictionary": self.dictionary,
				'num_frequencies': self.num_frequencies,	
				'embedding_length': self.num_embedding,
				'dropout': self.encoder_layer_args["dropout"]
			}],
		}
		network_params = {
			"arch_params" : arch_params,
		}

		return Network(**network_params).to(self.device), network_params
	def _configure_optimizer(self):
		decay_params = []
		nodecay_params = []

		for net in self.nets:
			for name, param in net.model.named_parameters():
				if not param.requires_grad:
					continue

				if param.dim() >= 2 and "embedding" not in name.lower():
					decay_params.append(param)
				else:
					nodecay_params.append(param)

		optim_groups = [
			{"params": decay_params, "weight_decay": self.optimizer_params["optimizer_args"]["weight_decay"]},
			{"params": nodecay_params, "weight_decay": 0.0},
		]

		del self.optimizer_params["optimizer_args"]["weight_decay"]
		
		return Optimizer(optim_groups, **self.optimizer_params)

	