import torch
import torch.nn as nn
import copy

from deeppy import BaseModel, Network, Optimizer

from .transformer import CausalMaskedTransformer
from .embedding import Embedding
from .vocab import HealthcareVocab

class DecoderLayer(nn.Module):
	def __init__(self, input_dim, output_dim, missing = False):
		super(DecoderLayer, self).__init__()
		self.missing = missing
		output_dim = output_dim if not missing else output_dim + 1
		self.decoder = nn.Sequential(
			nn.Linear(input_dim, input_dim * 2),
			nn.ReLU(),
			nn.Linear(input_dim * 2, output_dim)
		)
		self.reset_parameters()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.decoder(x)
		if not self.missing:
			return x
		if self.sigmoid(x[..., -1]) > 0.5:
			return torch.full_like(x[..., :-1], self.missing, device=x.device)
		return x[..., :-1]

	def reset_parameters(self):
		for m in self.decoder:
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)


class Decoder(nn.Module):
	def __init__(self, input_dim, dictionary):
		super(Decoder, self).__init__()
		self.input_dim = input_dim
		self.dictionary = dictionary

		self.column_decoders = DecoderLayer(input_dim, len(dictionary)) # Null, time
		self.categorical_decoder = DecoderLayer(input_dim * 2, len(dictionary)) # (column), (column,time)
		self.numeric_decoder = DecoderLayer(input_dim, 1, missing=torch.nan) # (column, time)
		self.time_decoder = DecoderLayer(input_dim, 1, missing=torch.inf) # (Null), (column), (column, code)

	def forward(self, X):
		# X shape: (batch_size, seq_len + 1, embedding_dim)
		# e shape: (batch_size, seq_len + 1, num_features, embedding_dim)
		# features (categorical, numeric, start_time, duration, dT, table_id, column_id, event_id) 
		x, e = X
		
		#discard first embedding, not going to be predicted
		e = e[..., 1:-1, :, :]
		#discard last embedding, not going to be predicted
		x = x[..., :-2, :]





class DBTransformer(BaseModel):
	dependencies = [Network, Optimizer]
	def __init__(self, dictionary_path, optimizer_params,
	 			num_frequencies=10,
				d_model = 64, nhead= 8, num_encoder_layers = 4, dim_feedforward = 256, 
				dropout = 0.1, activation = "gelu"):

		super(DBTransformer, self).__init__()
		self.optimizer_params = copy.deepcopy(optimizer_params)
		self.dictionary = HealthcareVocab(dictionary_path)  # Add this line to load the dictionary
		self.num_frequencies = num_frequencies

	
		self.encoder_layer_args = {"d_model": d_model, "nhead": nhead, "dim_feedforward": dim_feedforward, "dropout": dropout, "activation": activation}
		self.encoder_args = {"num_layers": num_encoder_layers}


	def forward(self, X):
		#T (Batch, Seq_len, Features)
		x, m = X[:2]
		e_raw, m = self.embedding_network(x)
		
		x = e_raw.sum(dim=2)  # Sum over the embedding dimension
		x = self.transformer_network((x, m))
		x = self.decoder_network((x, e_raw))

		return x
	
	def get_loss(self, X):
		X, y = X
		outs = self.forward(X)
		loss = self.criterion(outs, y)
		return loss

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

	def _init_loss_function(self):
		self.criterion = nn.MSELoss()
		self.loss_functions = [self.criterion]

	def _load_loss_function(self, loss_function):
		self.criterion = loss_function
	
	def _init_logger(self):
		tags = ["loss"]
		keys = [["loss"]]
		self.logger = self.create_logger(tags, keys)

	# =====================================================================
	def _build_decoder(self):
		arch_params = {
			"blocks" : [Decoder],
			"block_args" : [{
				"input_dim": self.encoder_layer_args["d_model"],
				"dictionary": self.dictionary
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
				"encoder_args": self.encoder_args
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
				'num_frequencies': self.num_frequencies
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

	