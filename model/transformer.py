import torch
import torch.nn as nn

class CausalMaskedTransformer(nn.Module):
	def __init__(self, encoder_layer_args, encoder_args):
		super(CausalMaskedTransformer, self).__init__()
		self.encoder_layer_args = {**encoder_layer_args,"batch_first": True}
		self.encoder_args = encoder_args

		# Define the transformer encoder layer
		encoder_layer = nn.TransformerEncoderLayer(**self.encoder_layer_args)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, **self.encoder_args)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_layer_args["d_model"]))

		self.apply(self._init_weights)
    
    def forward(self, X):
        x, padding_mask = X
		x = self.transformer_encoder(x, is_causal=True, src_key_padding_mask=padding_mask)
		return x

	def _init_weights(self, module):
		# 1. Handle all Linear layers (internal attention projections AND output heads)
		if isinstance(module, nn.Linear):
			# 0.02 is the universal magic number for transformer stability
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
				
		# 2. Handle embedding layers if you add them later
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			
		# 3. Handle Layer Normalization (Crucial: keep weight at 1, bias at 0)
		elif isinstance(module, nn.LayerNorm):
			nn.init.ones_(module.weight)
			nn.init.zeros_(module.bias)

		# 4. Handle PyTorch's native MultiheadAttention parameters explicitly
		elif isinstance(module, nn.MultiheadAttention):
			if module.in_proj_weight is not None:
				nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
			if module.q_proj_weight is not None:
				nn.init.normal_(module.q_proj_weight, mean=0.0, std=0.02)
			if module.k_proj_weight is not None:
				nn.init.normal_(module.k_proj_weight, mean=0.0, std=0.02)
			if module.v_proj_weight is not None:
				nn.init.normal_(module.v_proj_weight, mean=0.0, std=0.02)
			if module.in_proj_bias is not None:
				nn.init.zeros_(module.in_proj_bias)

		# 5. Initialize standalone parameters (like your CLS token)
		# Note: self.apply() catches sub-modules, we handle the raw Parameter explicitly inside __init__ or here
		if hasattr(self, 'cls_token') and self.cls_token is not None:
			# Match either 0.02 or your elegant d_model ** -0.5
			nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

