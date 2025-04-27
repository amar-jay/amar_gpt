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
		#TODO: apply dropout
		return self.fc2(F.relu(self.fc1(x)))

class AttentionType(Enum):
    CASUAL = 1
    GROUPED_QUERY = 2 #TODO: implement later
    LATENT = 3 #TODO: implement latent attention by deepseek

class Attention(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout_p=0.0, 
	      use_flash=False, 
			use_kv_cache = False,
			use_kv_norm=False):

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
		B, T, C = x.shape

		_k = self.k(x)
		_q = self.q(x)
		_v = self.v(x)

		if self.use_flash:
			masked = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=attention_mask, dropout_p=self.dropout_p)
		else: # else use vanilla attention mechanism
			batch_size, seq_len, channel_size = _k.shape
			if channel_size % self.num_heads != 0:
				raise ValueError("We have a fractional head_dim . the num_channels isn't perfectly divisible by head size")

			_k = _k.view(batch_size, seq_len, self.num_heads, channel_size // self.num_heads).transpose(1, 2)
			_q = _q.view(batch_size, seq_len, self.num_heads, channel_size // self.num_heads).transpose(1, 2)
			_v = _v.view(batch_size, seq_len, self.num_heads, channel_size // self.num_heads).transpose(1, 2)

			# kv norm
			kv = _k + _v
			_k /= kv
			_v /= kv

			att = F.softmax((_q @ _k.T) / _k.size(3))
			unmasked = att @ _v
			if attention_mask is None:
				attention_mask = torch.tril(torch.ones_like(unmasked))
			masked = unmasked * attention_mask
		masked = masked.transpose(1, 2).contiguous().view(B, T, C)
		#TODO: apply dropout here
		return masked


class PositionalEmbedding(nn.Module):
	def __init__(self, num_embeddings, vocab_size, emb_type="absolute", use_bias = False) -> None:
		super().__init__()
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
	      use_flash=True, dropout_p=0.0, use_kv_norm=True,
			num_att_blocks = 4
			):
		super().__init__()
		"""
		channel_size is the embed dimension of the KQV affine layers before atttended
		input_dim is basically the vocab_size, since it recievees input from the embedding layer directly
		"""
		self.num_heads = num_heads
		block = nn.Sequential(
			nn.LayerNorm(channel_size),
			Attention(input_dim, input_dim*4, dropout_p, 
		   use_flash=use_flash, use_kv_norm=use_kv_norm),
			nn.LayerNorm(channel_size),
			FeedForward(input_dim, input_dim, channel_size),
		)
		self.block_layers = nn.ModuleList([block for _ in range(num_att_blocks)])

	def forward(self, x):
		for _ in self.block_layers:
			x = self.block(x)
		return x


class Transformer(nn.Module):
	def __init__(self, vocab_size, dropout=0.3):
		super().__init__()
		self.emb = Embedding(8, vocab_size=vocab_size)
		self.dropout = nn.Dropout(dropout)
		self.att = MultiheadAttention(input_dim=vocab_size, channel_size=4*vocab_size, num_heads=4, num_att_blocks=1)
		self.norm = nn.LayerNorm(4*vocab_size)
		self.lm_head = nn.Linear(4*vocab_size, vocab_size, bias=False)

	def forward(self, x, targets=None):
		_emb = self.emb(x) # B, T, vocab_size
		x = self.att(_emb)
		x = self.dropout(x + _emb)
		x = self.norm(x)

		if targets is not None:
			logits = self.lm_head(x)
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
			return logits, loss

		# inference-time mini-optimization: only forward the lm_head on the very last position
		logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
		return logits, None


if __name__ == "__main__":
	# test the model
	model = Transformer(vocab_size=1000)
	x = torch.randint(0, 1000, (2, 10))
	logits, loss = model(x)
	print(logits.shape) # should be (2, 1, 1000)
	print(loss) # should be None