import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum



class FeedForward(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim):
		super(FeedForward, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		#TODO: apply a better activation function
		#TODO: apply MoE later
		return self.fc2(F.relu(self.fc1(x)))

class AttentionType(Enum):
    CASUAL = 1
    GROUPED_QUERY = 2 #TODO: implement later
    LATENT = 3 #TODO: implement latent attention by deepseek

class Attention(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout_p=0.0, use_flash=False):
		super(Attention, self).__init__()
		self.k = nn.Linear(input_dim, hidden_dim)
		self.q = nn.Linear(input_dim, hidden_dim)
		self.v = nn.Linear(input_dim, hidden_dim)
		self.dropout_p = dropout_p

		check_flash = hasattr(F, "scaled_dot_product_attention")
		if use_flash and not check_flash:
			raise ValueError("Use a better pytorch version to use flash attention ie >2.0")
		self.use_flash = True


	def forward(self, x, attention_mask=None):

		_k = self.k(x)
		_q = self.q(x)
		_v = self.v(x)

		if self.use_flash:
			masked = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=attention_mask, dropout_p=self.dropout_p)
		else: # else use vanilla attention mechanism
			batch_size, seq_len, channel_size = _k.shape
			if channel_size % self.num_heads != 0:
				raise ValueError("We have a fractional head_dim . the num_channels isn't perfectly divisible by head size")

			_k = _k.resize(batch_size, seq_len, self.num_heads, channel_size)
			_q = _q.resize(batch_size, seq_len, self.num_heads, channel_size)
			_v = _v.resize(batch_size, seq_len, self.num_heads, channel_size)

			att = F.softmax((_q @ _k.T) / _k.size(3))
			unmasked = att @ _v
			if attention_mask is None:
				attention_mask = torch.tril(torch.ones_like(unmasked))
			masked = unmasked * attention_mask
		return masked


class PositionalEmbedding(nn.Module):
	def __init__(self, num_embeddings, vocab_size, emb_type="absolute", use_bias = False) -> None:
		self.emb_type = emb_type
		self.weight = nn.Parameter(torch.ones(vocab_size, num_embeddings))
		if use_bias:
			self.bias = nn.Parameter(torch.zeros(num_embeddings))

	def forward(self, x):
		if self.emb_type == "absolute":
			seq_len = x.size(2)
			_pos = torch.arange(0, seq_len)
			return _pos @ self.weight.T
		else:
			raise TypeError(f"do not support {self.emb_type=} of positional embedding")

class Embedding(nn.Module):
	def __init__(self, num_embeddings, vocab_size) -> None:
		super().__init__()
		self.token_embedding = nn.Embedding(num_embeddings, vocab_size) # text embedding
		self.positional_embedding = PositionalEmbedding(num_embeddings, vocab_size)# positional embedding, there are many types TODO: try to implement that of Gemma paper
	def forward(self, x):
		tok = self.token_embedding(x) # B, T, vocab_size
		pos = self.positional_embedding(x)
		return tok + pos


class MultiheadAttention(nn.Module):
	def __init__(self, input_dim, channel_size, num_heads=4, 
	      use_self_attention=True, use_dropout=True, use_kv_norm=True):
		"""
		channel_size is the embed dimension of the KQV affine layers before atttended
		input_dim is basically the vocab_size, since it recievees input from the embedding layer directly
		"""
		head_dim =  channel_size // num_heads
		self.num_heads = num_heads
		self.k = nn.Linear(input_dim, channel_size) 
		self.q = nn.Linear(input_dim, channel_size)
		self.v = nn.Linear(input_dim, channel_size)

	def forward(self, x):
		_k = self.k(x)
		_q = self.q(x)
		_v = self.v(x)

		batch_size, seq_len, channel_size = _k.shape
		if channel_size % self.num_heads != 0:
			raise ValueError("We have a fractional head_dim . the num_channels isn't perfectly divisible by head size")
		_k = _k.view(batch_size, seq_len, self.num_heads, channel_size // self.num_heads).transpose(1, 2)
		_q = _q.view(batch_size, seq_len, self.num_heads, channel_size // self.num_heads).transpose(1, 2)
		_v = _v.view(batch_size, seq_len, self.num_heads, channel_size // self.num_heads).transpose(1, 2)

		# att = (_k @ _q.T) / (_k.size(2) ** 0.5)
		# result = F.softmax(att) @ _v


class Transformer(nn.Module):
	def __init__(self, vocab_size, dropout=0.3):
		super().__init__()
		self.emb = Embedding(8, vocab_size=vocab_size)
		self.att = MultiheadAttention(input_dim=vocab_size, channel_size=4*vocab_size, num_heads=4)
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(vocab_size)
		self.ff = FeedForward(vocab_size, vocab_size, 4*vocab_size, )

	def forward(self, x):
		_emb = self.emb(x) # B, T, vocab_size
		_att = self.att(_emb)
		_att = self.dropout(_att + _emb)
		_att = self.norm(_att)
		_ff = self.ff(_att)

		return _ff

