"""Standalone attention variants for learning and shape inspection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _init_linear(layer: nn.Linear, scale: float) -> None:
	nn.init.normal_(layer.weight, mean=0.0, std=scale)
	nn.init.zeros_(layer.bias)


def _split_heads(values: Tensor, heads: int) -> Tensor:
	batch_size, token_count, hidden_size = values.shape
	head_dim = hidden_size // heads
	return values.view(batch_size, token_count, heads, head_dim).transpose(1, 2)


def _merge_heads(values: Tensor) -> Tensor:
	batch_size, heads, token_count, head_dim = values.shape
	return values.transpose(1, 2).contiguous().view(batch_size, token_count, heads * head_dim)


def _repeat_kv(values: Tensor, repeats: int) -> Tensor:
	batch_size, kv_heads, token_count, head_dim = values.shape
	values = values[:, :, None, :, :]
	values = values.expand(batch_size, kv_heads, repeats, token_count, head_dim)
	return values.reshape(batch_size, kv_heads * repeats, token_count, head_dim)


class CrossAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.head_dim = hidden_size // heads
		self.q_proj = nn.Linear(hidden_size, hidden_size)
		self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size)
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.q_proj, self.kv_proj, self.output_proj):
			_init_linear(layer, scale)

	def _split_heads(self, values: Tensor) -> Tensor:
		return _split_heads(values, self.heads)

	def _merge_heads(self, values: Tensor) -> Tensor:
		return _merge_heads(values)

	def forward(self, hidden_states: Tensor, context_states: Tensor) -> Tensor:
		q = self.q_proj(hidden_states)
		k, v = self.kv_proj(context_states).chunk(2, dim=-1)
		q = self._split_heads(q, self.heads)
		k = self._split_heads(k, self.heads)
		v = self._split_heads(v, self.heads)
		output = f.scale_dot_product_attention(q, k, v)
		output = self.output_proj(self._merge_heads(output))
		return self.dropout(output)


##############################################################################################
#                                            MHA                                             #
##############################################################################################


class MultiHeadAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.qkv, self.output_proj):
			_init_linear(layer, scale)

	def forward(self, hidden_states: Tensor) -> Tensor:
		q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
		q = _split_heads(q, self.heads)
		k = _split_heads(k, self.heads)
		v = _split_heads(v, self.heads)
		output = F.scale_dot_product_attention(q, k, v)
		output = self.output_proj(_merge_heads(output))
		output = self.dropout(output)
		return output

##############################################################################################
#                                            GQA                                             #
##############################################################################################


class GroupedQueryAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, kv_heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.kv_heads = kv_heads
		self.head_dim = hidden_size // heads
		self.q_proj = nn.Linear(hidden_size, hidden_size)
		self.kv_proj = nn.Linear(hidden_size, 2 * kv_heads * self.head_dim)
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.q_proj, self.kv_proj, self.output_proj):
			_init_linear(layer, scale)

	def forward(self, hidden_states: Tensor) -> Tensor:
		q = _split_heads(self.q_proj(hidden_states), self.heads)
		k, v = self.kv_proj(hidden_states).chunk(2, dim=-1)
		k = _split_heads(k, self.kv_heads)
		v = _split_heads(v, self.kv_heads)
		repeats = self.heads // self.kv_heads
		k = _repeat_kv(k, repeats)
		v = _repeat_kv(v, repeats)
		output = f.scale_dot_product_attention(q, k, v)
		output = self.output_proj(_merge_heads(output))
		return self.dropout(output)


##############################################################################################
#                                           MLA                                              #
##############################################################################################


class MultiHeadLatentAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, latent_size: int | None = None, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.latent_size = latent_size
		self.q_proj = nn.Linear(hidden_size, latent_size)
		self.kv_proj = nn.Linear(hidden_size, 2 * latent_size)
		self.output_proj = nn.Linear(latent_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.q_proj, self.kv_proj, self.output_proj):
			_init_linear(layer, scale)

	def forward(self, hidden_states: Tensor) -> Tensor:
		
		q = self.q_proj(hidden_states)
		k, v = self.kv_proj(hidden_states).chunk(2, dim=-1)
		q = _split_heads(q, self.heads)
		k = _split_heads(k, self.heads)
		v = _split_heads(v, self.heads)
		output = F.scale_dot_product_attention(q, k, v)
		output = self.output_proj(_merge_heads(output))
		return self.dropout(output)



##############################################################################################
#                                            SWA                                             #
##############################################################################################


class SlidingWindowAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, window_size: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.window_size = window_size
		self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.qkv, self.output_proj):
			_init_linear(layer, scale)

	def _mask(self, token_count: int, device: torch.device) -> Tensor:
		positions = torch.arange(token_count, device=device)
		return (positions[:, None] - positions[None, :]).abs() < self.window_size

	def forward(self, hidden_states: Tensor) -> Tensor:
		q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
		q = _split_heads(q, self.heads)
		k = _split_heads(k, self.heads)
		v = _split_heads(v, self.heads)
		mask = self._mask(hidden_states.shape[1], hidden_states.device)
		output = F.scale_dot_product_attention(q, k, v)
		output = self.output_proj(_merge_heads(output))
		return self.dropout(output)


##############################################################################################
#                                            SPA                                             #
##############################################################################################


class SparseAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.qkv, self.output_proj):
			_init_linear(layer, scale)

	def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:

		q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
		q = _split_heads(q, self.heads)
		k = _split_heads(k, self.heads)
		v = _split_heads(v, self.heads)
		output = F.scale_dot_product_attention(q, k, v)
		output = self.output_proj(_merge_heads(output))
		return self.dropout(output)


##############################################################################################
#                                     Linear-Attention                                       #
##############################################################################################


class LinearAttention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.qkv, self.output_proj):
			_init_linear(layer, scale)

	def forward(self, hidden_states: Tensor) -> Tensor:
		q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
		q = _split_heads(q, self.heads)
		k = _split_heads(k, self.heads)
		v = _split_heads(v, self.heads)
		q = F.elu(q) + 1.0
		k = F.elu(k) + 1.0
		context = torch.einsum("bhtd,bhte->bhde", k, v)
		normalizer = torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2)).clamp_min(1e-6)
		output = torch.einsum("bhtd,bhde->bhte", q, context) / normalizer.unsqueeze(-1)
		output = self.output_proj(_merge_heads(output))
		output = self.dropout(output)
		return output



class Attention(nn.Module):
	def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.heads = heads
		# self.kv_heads = kv_heads in "GQA"
		self.head_dim = hidden_size // heads
		self.q_proj = nn.Linear(hidden_size, hidden_size)
		self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size)
		# self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size) in "MHSA"
		self.output_proj = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(dropout)

		scale = 1.0 / math.sqrt(hidden_size)
		for layer in (self.q_proj, self.kv_proj, self.output_proj):
			_init_linear(layer, scale)

	def _split_heads(self, values: Tensor) -> Tensor:
		return _split_heads(values, self.heads)

	def _merge_heads(self, values: Tensor) -> Tensor:
		return _merge_heads(values)

	def forward(self, hidden_states: Tensor, context_states: Tensor) -> Tensor:
		q = self.q_proj(hidden_states)
		k, v = self.kv_proj(context_states).chunk(2, dim=-1)
		q = self._split_heads(q, self.heads)
		k = self._split_heads(k, self.heads)
		v = self._split_heads(v, self.heads)
		output = f.scale_dot_product_attention(q, k, v)
		output = self.output_proj(self._merge_heads(output))
		return self.dropout(output)