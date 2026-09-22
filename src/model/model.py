import torch
import torch.nn as nn
import copy

from deeppy import BaseModel, Network, Optimizer
from . import CausalMaskedTransformer, Embedding, \
	QuestionDecoder, GaussianMixtureEstimator, StatLogger

from .. import HealthCareDictionary

class DBTransformer(BaseModel):
	dependencies = [Network, Optimizer]
	def __init__(self, dictionary_path, optimizer_params,
				num_frequencies=10, num_questions=6, num_embedding=6, num_gaussians=10,
				d_model = 64, nhead= 8, num_encoder_layers = 4, dim_feedforward = 256, 
				context_size = 2048, dropout = 0.1, activation = "gelu", 
				last_layer_norm = True, use_checkpointing=False):

		super(DBTransformer, self).__init__()
		self.optimizer_params = copy.deepcopy(optimizer_params)
		self.dictionary = HealthCareDictionary(dictionary_path)  # Add this line to load the dictionary
		
		
		self.num_frequencies = num_frequencies
		self.num_questions = num_questions
		self.num_embedding = num_embedding
		self.use_checkpointing = use_checkpointing
		self.num_gaussians = num_gaussians
		self.question_dim = d_model // num_embedding
		
		self.context_size = context_size
		self.encoder_layer_args = {"d_model": d_model, "nhead": nhead, "dim_feedforward": dim_feedforward, "dropout": dropout, "activation": activation}
		self.encoder_args = {"num_layers": num_encoder_layers, "norm": nn.LayerNorm(d_model) if last_layer_norm else None}

	def forward(self, X):
		#T (Batch, Seq_len, Features)
		x, m = X[:2]
		tokens, e_raw, m = self.embedding_network((x,m))
		
		latent = self.transformer_network((tokens, m))
		return self.decoder_network((latent, e_raw)) #class_logits, time_params, code_logits, value_params
	
	def get_loss(self, X):
		x, m, labels = X
		class_logits, time_params_none, time_params_table, time_params_code, code_logits = self.forward((x,m))

		loss = self.compute_loss(x, m, labels, class_logits, time_params_none, time_params_table, time_params_code, code_logits)
		return loss

	#padding_mask shape = (batch_size, context length)
	#labels shape = (batch_size, context length)
	def compute_loss(self, x, padding_mask, labels, class_logits, time_params_none, time_params_table, time_params_code, code_logits):
		class_loss = self.classification_task(class_logits, labels, padding_mask)

		reconstruction_mask = padding_mask.bool() | labels.eq(0)
		code_loss = self.multilabel_classification_task(code_logits, x[..., 0, :], reconstruction_mask)
		time_loss_none = self.gaussian_task(time_params_none, x[..., 3, :], "Time None", reconstruction_mask)  # Start time
		time_loss_table = self.gaussian_task(time_params_table, x[..., 3, :], "Time Table", reconstruction_mask)  # Start time
		time_loss_code = self.gaussian_task(time_params_code, x[..., 3, :], "Time Code", reconstruction_mask)  # Start time
		self.stat_logger.log_loss(torch.stack([class_loss, time_loss_none, time_loss_table, time_loss_code, code_loss]))
		return class_loss + time_loss_none + time_loss_table + time_loss_code + code_loss
	#logits shape = (batch_size, context length, 1, 1)
	#y shape = (batch_size, context length)
	def classification_task(self, class_logits, y, padding_mask):
		padding_mask = padding_mask.clone()
		padding_mask[:, 0] = 1  # No classification loss for the first token
		class_logits = class_logits.squeeze(-1).squeeze(-1)  # Shape: (batch_size, context length)
		loss = self.bce_loss(class_logits, y)
		self.stat_logger.log_classification(class_logits[~padding_mask.bool()].detach(), y[~padding_mask.bool()].detach(), "Classification", task="binary")
		
		return loss[~padding_mask.bool()].mean()  

	#codelogits shape = (batch_size, context length, dictionary_size)
	#y shape = (batch_size, context length)
	#padding_mask shape = (batch_size, context length)
	def multilabel_classification_task(self, code_logits, y, padding_mask):
		loss = self.cross_entropy_loss(code_logits.squeeze(-2).transpose(1, 2), y.long())
		self.stat_logger.log_classification(code_logits[~padding_mask.bool()].detach(), y[~padding_mask.bool()].detach(), "Code", task="multiclass")
		return loss[~padding_mask.bool()].mean()

	#estimation shape = (batch_size, context length,1, 3 * n_gaussians)
	#y shape = (batch_size, context length)
	def gaussian_task(self, estimation, y, log_name, padding_mask):
		log_likelihoods = GaussianMixtureEstimator.log_likelihood(y, estimation.squeeze(-2), self.num_gaussians)
		return -log_likelihoods[~padding_mask.bool()].mean()


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
		tags = ["Loss", "Classification", "Code Top-K Accuracy", "Code Top-K AUC", "Code Entropy"]
		keys = [["Classification Loss", "Time Loss None", "Time Loss Table" , "Time Loss Code", "Code Loss"],
				["AUC-ROC", "Precision", "Recall", "Accuracy","Entropy"],
				["Top-1", "Top-5", "Top-10", "Top-20"],
				["Top-1", "Top-5", "Top-10", "Top-20"],
				["Entropy"],
		]
		self.logger = self.create_logger(tags, keys)
		self.stat_logger = StatLogger(self.logger, num_classes=len(self.dictionary), device=self.device)
	

	# =====================================================================
	def _build_decoder(self):
		arch_params = {
			"blocks" : [QuestionDecoder],
			"block_args" : [{
				"embedding_dim": self.encoder_layer_args["d_model"],
				"question_dim": self.question_dim,
				"dictionary_size": len(self.dictionary),
				"num_questions": self.num_questions,
				"dropout": self.encoder_layer_args["dropout"],
				"n_gaussians": self.num_gaussians
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

	