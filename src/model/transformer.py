import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SDPATransformerEncoderLayer(nn.TransformerEncoderLayer):
	"""Transformer encoder layer whose attention path explicitly uses SDPA."""

	def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
		# Retain TransformerEncoderLayer's packed projections for checkpoint
		# compatibility while making the attention kernel explicit.
		q, k, v = F.linear(x, self.self_attn.in_proj_weight,
			self.self_attn.in_proj_bias).chunk(3, dim=-1)
		batch_size, sequence_length, _ = q.shape
		num_heads = self.self_attn.num_heads
		head_dim = self.self_attn.head_dim

		q = q.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
		k = k.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
		v = v.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

		# The data is right-padded and attention is causal, so padding tokens are
		# always in the future of every valid query. Ignoring the padding mask
		# keeps attn_mask=None, which is required by the FlashAttention kernel.
		sdpa_mask = None
		if key_padding_mask is not None and not is_causal:
			sdpa_mask = ~key_padding_mask[:, None, None, :].bool()

		if attn_mask is not None:
			if attn_mask.dtype == torch.bool:
				attn_mask = ~attn_mask
			if attn_mask.dim() == 2:
				attn_mask = attn_mask[None, None, :, :]
			if sdpa_mask is None:
				sdpa_mask = attn_mask
			elif attn_mask.dtype == torch.bool:
				sdpa_mask = sdpa_mask & attn_mask
			else:
				sdpa_mask = attn_mask.masked_fill(~sdpa_mask, float("-inf"))

		attention = F.scaled_dot_product_attention(
			q, k, v,
			attn_mask=sdpa_mask,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=is_causal and sdpa_mask is None,
		)
		attention = attention.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
		attention = self.self_attn.out_proj(attention)
		return self.dropout1(attention)


class CausalMaskedTransformer(nn.Module):
	def __init__(self, context_size, encoder_layer_args, encoder_args, use_checkpointing=False):
		super(CausalMaskedTransformer, self).__init__()
		self.encoder_layer_args = {**encoder_layer_args,"batch_first": True}
		self.encoder_args = encoder_args
		self.use_checkpointing = use_checkpointing

		# Define the transformer encoder layer
		encoder_layer = SDPATransformerEncoderLayer(**self.encoder_layer_args)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, **self.encoder_args)
		#self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_layer_args["d_model"]))

		self.apply(self._init_weights)

		self.register_buffer(
			"causal_mask",
			torch.triu(
				torch.ones(context_size + 1, context_size + 1, dtype=torch.bool),diagonal=1
			)
		)

	def forward(self, X):
		if self.use_checkpointing:
			return self.forward_checkpointed(X)
		else:
			return self.forward_normal(X)

	def forward_checkpointed(self, X):
		x, padding_mask = X

		# Iterate through encoder layers manually
		for layer in self.transformer_encoder.layers:

			def layer_forward(x, layer=layer):
				return layer(x, src_key_padding_mask=padding_mask, is_causal=True)

			x = checkpoint(layer_forward, x, use_reentrant=False)
		# TransformerEncoder can have a final LayerNorm
		if self.transformer_encoder.norm is not None:
			x = self.transformer_encoder.norm(x)

		return x
	def forward_normal(self, X):
		x, padding_mask = X
		x = self.transformer_encoder(x,src_key_padding_mask=padding_mask,is_causal=True)
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

