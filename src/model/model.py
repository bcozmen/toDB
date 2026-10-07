import torch
import torch.nn as nn
import copy

from deeppy import BaseModel, Network, Optimizer
from . import CausalMaskedTransformer, Embedding, \
	PredictionDecoder, GaussianMixtureEstimator, StatLogger
from torch.nn.attention import SDPBackend, sdpa_kernel

from .. import HealthCareDictionary

IMPORTANT_ENCOUNTER_INDICES = [984, 1034, 1016, 1036, 1037, 1038, 1043, 1044, 990, 991, 992, 995, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1014, 1021, 1024, 1033, 1041, 1042, 986]
# Time is normalized by ten years in the dataset loader.
FUTURE_ENCOUNTER_HORIZON = 3650 / 3650  # 10 years


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
		self.future_horizon = FUTURE_ENCOUNTER_HORIZON
		
		self.context_size = context_size
		self.encoder_layer_args = {"d_model": d_model, "nhead": nhead, "dim_feedforward": dim_feedforward, "dropout": dropout, "activation": activation}
		self.encoder_args = {"num_layers": num_encoder_layers, "norm": nn.LayerNorm(d_model) if last_layer_norm else None}

	def forward(self, X):
		#T (Batch, Seq_len, Features)
		x, m = X
		tokens, e_raw, m = self.embedding_network((x,m))
		
		#with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
		#	latent = self.transformer_network((tokens, m))
		latent = self.transformer_network((tokens, m))
		return self.decoder_network((latent, x))
	
	def get_loss(self, X):
		x, m, labels = X
		class_logits, next_event, future_encounter = self.forward((x, m))

		loss = self.classification_task(class_logits, labels, m)
		next_losses = self.next_event_losses(x, m, labels, *next_event) 
		future_losses = self.future_encounter_losses(x, m, labels, *future_encounter)

		total_loss = loss + next_losses.sum() + future_losses.sum()
		losses = torch.cat((loss.unsqueeze(0), next_losses, future_losses))
		self.stat_logger.log_entropy()
		self.stat_logger.log_loss(losses)
		return total_loss

	def next_event_losses(self, x, padding_mask, labels, time_params, table_logits, code_logits):
		"""Teacher-forced loss for time, table, and code prediction."""
		valid = ~(padding_mask.bool() | labels.eq(0))
		valid = valid & (x[..., 8, :] > 0)
		time = torch.nan_to_num(x[..., 5, :], nan=0.0, posinf=0.0, neginf=0.0)
		table = x[..., 6, :].long()
		code = x[..., 0, :].long()

		time_loss = -GaussianMixtureEstimator.log_likelihood(time, time_params.squeeze(-2), self.num_gaussians)
		table_loss = self.cross_entropy_loss(table_logits.transpose(1, 2), table)
		code_loss = self.cross_entropy_loss(code_logits.transpose(1, 2), code)
		self.stat_logger.log_gaussian(time_params.squeeze(-2)[valid], time[valid], "Next Time", self.num_gaussians)
		
		self._log_next_event_metrics(table_logits, code_logits, table, code, valid)
		return torch.stack((time_loss[valid].mean(), table_loss[valid].mean(), code_loss[valid].mean())) if valid.any() else torch.zeros(3, device=x.device)

	def future_encounter_losses(self, x, padding_mask, labels, future_params, code_logits):
		"""Predict the closest future important encounter from each encounter.

		The time target is measured from the prediction time immediately before
		the current sequence position to the target encounter. Channel 3 contains
		birth-relative age, so the target is the difference between target and
		prediction ages.
		"""
		observed = ~(padding_mask.bool() | labels.eq(0))
		encounters = observed & x[..., 8, :].eq(1)
		ages = torch.nan_to_num(x[..., 3, :], nan=0.0, posinf=0.0, neginf=0.0)
		codes = x[..., 0, :].long()

		important = torch.isin(codes, x.new_tensor(IMPORTANT_ENCOUNTER_INDICES).long()) & observed
		positions = torch.arange(x.size(-1), device=x.device).expand_as(codes)
		suffix_positions = torch.flip(
			torch.cummin(torch.flip(torch.where(important, positions, x.size(-1)), (1,)), 1).values,
			(1,),
		)
		has_target = suffix_positions.lt(x.size(-1))
		target_positions = suffix_positions.clamp_max(x.size(-1) - 1)


		prediction_positions = (positions - 1).clamp_min(0)
		prediction_ages = ages.gather(1, prediction_positions)
		target_ages = ages.gather(1, target_positions)
		target_times = (target_ages - prediction_ages).clamp_min(0.0)
		target_codes = codes.gather(1, target_positions)
		encounters = encounters & has_target

		if not encounters.any():
			self.stat_logger.log_gaussian(future_params[encounters, :], target_times[encounters], "Future Time", self.num_gaussians)
			return torch.zeros(2, device=x.device)

		future_time_loss = -GaussianMixtureEstimator.log_likelihood(
			target_times[encounters], future_params[encounters, :], self.num_gaussians
		)
		self.stat_logger.log_gaussian(
			future_params[encounters, :], target_times[encounters], "Future Time", self.num_gaussians
		)
		code_loss = self.cross_entropy_loss(code_logits[encounters, :], target_codes[encounters])

		self.stat_logger.log_classification(
			code_logits[encounters].detach(), target_codes[encounters], "Future Code", task="multiclass"
		)
		return torch.stack((future_time_loss.mean(), code_loss.mean()))
		

	def _log_next_event_metrics(self, table_logits, code_logits, table, code, valid):
		self.stat_logger.log_classification(
			table_logits[valid].detach(), table[valid], "Next Table", task="multiclass"
		)
		self.stat_logger.log_classification(
			code_logits[valid].detach(), code[valid], "Next Code", task="multiclass"
		)

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
			"Loss", "Classification", 
			"Next Time", "Future Time",
			"Next Table Top-K Accuracy", "Next Table Top-K AUC",
			"Next Code Top-K Accuracy", "Next Code Top-K AUC",
			"Future Code Top-K Accuracy", "Future Code Top-K AUC", "Entropy",
		]
		keys = [
			["Classification Loss", "Next Time Loss", "Next Table Loss", "Next Code Loss",
			 "Future Time Loss", "Future Code Loss"],
			["AUC-ROC", "Precision", "Recall", "Accuracy"],
			["MAE", "RMSE"],
			["MAE", "RMSE"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Top-1", "Top-5", "Top-10", "Top-20"],
			["Classification", "Next Table", "Next Code", "Future Time", "Future Code"],
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

	def load(self, file_name):
		print(f"Loading model from {file_name}")
		if isinstance(file_name, dict):
			checkpoint = file_name
		else:
			checkpoint = torch.load(file_name + "/model.pt", weights_only = False)
		print(f"checkpoint keys: {list(checkpoint.keys())}")
		params = checkpoint["params"]
		network_dicts = checkpoint["nets"]
		print(f"network_dicts keys: {list(network_dicts[0].keys())}")
		loss_functions = checkpoint["loss_functions"]
		optimizer_dicts = checkpoint["optimizer"]

		for net, net_dict in zip(self.nets, network_dicts):
			net.load_states(net_dict)

		if optimizer_dicts is not None:
			for optimizer, optimizer_dict in zip(self.optimizers, optimizer_dicts):
				optimizer.load_states(optimizer_dict)


		self._load_loss_functions(loss_functions)
	