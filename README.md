# Transformer Summarizer

A transformer-based text summarization project implementing text preprocessing pipelines and core Transformer architecture components inspired by the paper:

> *Attention Is All You Need* (Vaswani et al., 2017)

The repository focuses on understanding how Transformer models process text through tokenization, embeddings, positional encoding, self-attention, encoder-decoder structures, and sequence generation.

# Tech Stack

`PyTorch` `NumPy` `Pandas` `Matplotlib` `NLTK` `Jupyter Notebook`

# Project

## Transformer-Based Text Summarization

### Overview

Implemented preprocessing pipelines and Transformer architecture components for abstractive text summarization tasks.

The project explores how attention mechanisms and encoder-decoder architectures process natural language for sequence-to-sequence learning.

### Key Concepts

* Text preprocessing
* Tokenization
* Vocabulary building
* Positional encoding
* Self-attention
* Multi-head attention
* Encoder-decoder architecture
* Transformer blocks
* Sequence-to-sequence modeling

### Technologies Used

`PyTorch` `NLTK` `NumPy` `Pandas`

### Transformer Structure

<details open>
  <summary>Architecture</summary>

<img width="396" height="560" alt="image" src="https://github.com/user-attachments/assets/c130cf62-1b97-4ed6-859d-19fa5ec45e90" />

</details>

### Encoder Layer

<details>
  <summary>Code</summary>

```python
class EncoderLayer(nn.Module):
  def __init__(
    self,
    d_model,
    num_heads,
    d_ff,
    dropout_rate,
    bias,
    activation_type,
    eps=1e-5,
    elementwise_affine=True
    ):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_ff = d_ff
    self.dropout_rate = dropout_rate
    self.eps = eps
    self.bias = bias
    self.activation_type = activation_type

    self.attention = MultiHeadAttention(
      d_model=self.d_model,
      num_heads=self.num_heads,
      dropout_rate=self.dropout_rate,
      bias=self.bias
    )
    self.feed_forward = FeedForward(
      d_model=self.d_model,
      d_ff=self.d_ff,
      bias=self.bias,
      activation_type=self.activation_type,
      dropout_rate=self.dropout_rate
    )
    self.add_norm_1 = AddNorm(
      d_model=self.d_model,
      dropout_rate=self.dropout_rate,
      elementwise_affine=elementwise_affine,
      eps=self.eps
    )
    self.add_norm_2 = AddNorm(
      d_model=self.d_model,
      dropout_rate=self.dropout_rate,
      elementwise_affine=elementwise_affine,
      eps=self.eps
    )

  def forward(self, x, src_mask):
    attn_out, _ = self.attention(x, x, x, mask=src_mask)
    x = self.add_norm_1(x, attn_out)
    ffn_out = self.feed_forward(x)
    x = self.add_norm_2(x, ffn_out)
    return x
```

</details>

### Decoder Layer

<details>
  <summary>Code</summary>

```python
class DecoderLayer(nn.Module):
  def __init__(
    self,
    d_model,
    num_heads,
    d_ff,
    dropout_rate,
    bias,
    activation_type,
    eps=1e-5,
    elementwise_affine=True
    ):
    super().__init__()

    self.d_model = d_model
    self.dropout_rate = dropout_rate
    self.eps = eps

    self.self_attention = MultiHeadAttention(
      self.d_model,
      num_heads,
      self.dropout_rate,
      bias
    )

    self.cross_attention = MultiHeadAttention(
      self.d_model,
      num_heads,
      self.dropout_rate,
      bias
    )

    self.feed_forward = FeedForward(
      self.d_model,
      d_ff,
      bias,
      activation_type,
      self.dropout_rate
    )

    self.norm1 = AddNorm(
      self.d_model,
      self.dropout_rate,
      elementwise_affine,
      self.eps
    )

    self.norm2 = AddNorm(
      self.d_model,
      self.dropout_rate,
      elementwise_affine,
      self.eps
    )

    self.norm3 = AddNorm(
      self.d_model,
      self.dropout_rate,
      elementwise_affine,
      self.eps
    )

  def forward(self, y, encoder_output, src_mask, tgt_mask):
    self_attn_out, _ = self.self_attention(y, y, y, mask=tgt_mask)
    y = self.norm1(y, self_attn_out)

    cross_attn_out, _ = self.cross_attention(
      y,
      encoder_output,
      encoder_output,
      mask=src_mask
    )
    y = self.norm2(y, cross_attn_out)

    ffn_out = self.feed_forward(y)
    y = self.norm3(y, ffn_out)

    return y
```

</details>

### Attention Mechanism

#### Scaled Dot-Product Attention

<details open>
  <summary>Architecture</summary>

<img width="204" height="264" alt="image" src="https://github.com/user-attachments/assets/2e57839a-dad3-4156-a193-d54f4143cbc9" />

</details>

#### Multi-Head Attention

<details open>
  <summary>Architecture</summary>

<img width="242" height="302" alt="image" src="https://github.com/user-attachments/assets/6ce0d7ce-603a-4c39-a7f8-395cdab40682" />

</details>

#### Attention Implementation

<details>
  <summary>Code</summary>

```python
class MultiHeadAttention(nn.Module):
  def __init__(
      self,
      d_model: int,
      num_heads: int,
      dropout_rate: float,
      bias: bool
  ):
    super().__init__()

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.d_model = d_model
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.bias = bias

    self.d_k = self.d_model // self.num_heads
    self.scale = 1 / math.sqrt(self.d_k)

    self.q_proj = nn.Linear(self.d_model, self.d_model, self.bias)
    self.k_proj = nn.Linear(self.d_model, self.d_model, self.bias)
    self.v_proj = nn.Linear(self.d_model, self.d_model, self.bias)
    self.out_proj = nn.Linear(self.d_model, self.d_model, self.bias)

    self.dropout_layer = nn.Dropout(self.dropout_rate)

  def forward(self, Q, K, V, mask=None):
    batch, Lq, _ = Q.shape # (batch, seq_len, d_model)
    _, Lk, _ = K.shape
    # -> ( batch, seq_len, d_model ) ->
    # -> ( batch, seq_len, num_heads, d_k ) ->
    # -> ( batch, num_heads, seq_len, d_k )
    Q_heads = self.q_proj(Q).reshape(batch, Lq, self.num_heads, self.d_k).transpose(1, 2)
    K_heads = self.k_proj(K).reshape(batch, Lk, self.num_heads, self.d_k).transpose(1, 2)
    V_heads = self.v_proj(V).reshape(batch, Lk, self.num_heads, self.d_k).transpose(1, 2)

    # attention scores
    scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))

    # scaling
    scores = scores * self.scale

    # masking
    mask = self.validate_mask(mask, scores)
    if mask is not None:
      scores = scores + mask

    # attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = self.dropout_layer(attn_weights)

    # weighted sum of values
    attn_output = torch.matmul(attn_weights, V_heads).transpose(1,2).contiguous().reshape(batch, Lq, self.d_model)

    # output
    output = self.out_proj(attn_output)

    return output, attn_weights # optional return

  def validate_mask(self, mask, scores):
    if mask is None:
        return None

    # Check dimensions
    assert mask.dim() == 4, "Mask must be 4D (batch, heads/1, Lq, Lk)"

    B, H, Lq, Lk = scores.shape
    mB, mH, mLq, mLk = mask.shape

    # Shape compatibility
    assert mLk == Lk, "Mask last dim must match key length"
    assert mLq in (1, Lq), "Mask query dim must be 1 or Lq"
    assert mB in (1, B), "Mask batch dim must be 1 or batch size"
    assert mH in (1, H), "Mask head dim must be 1 or num_heads"

    # Dtype check
    assert mask.dtype.is_floating_point, "Mask must be float (additive mask)"

    # Device alignment
    mask = mask.to(device=scores.device, dtype=scores.dtype)

    return mask
```

</details>

### Positional Encoding

<details>
  <summary>Code</summary>

```python
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_len):
    super().__init__()

    pe = np.zeros((max_seq_len, d_model))
    positions = np.arange(max_seq_len)[:, None]

    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)

    self.register_buffer("pe", torch.tensor(pe, dtype=torch.float32))

  def forward(self, x):
    seq_len = x.shape[1]
    return x + self.pe[:seq_len].unsqueeze(0)
```

</details>
