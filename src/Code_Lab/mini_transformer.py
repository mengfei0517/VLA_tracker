import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
# 1. Scaled Dot-Product Attention (chapter 3.2.1)
# ---------------------------------------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Corresponding to the formula in the paper: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = q.size(-1)
    
    # Calculate QK^T and scale (Scaling)
    # Purpose: Prevent the dot product from being too large when d_k is large, causing the softmax gradient to vanish
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Mask processing (Padding Mask)
    # Set the position where mask is 0 to a very small value, and the weight after Softmax is almost 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Normalization (Softmax)
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum to get the output
    return torch.matmul(attn_weights, v), attn_weights


# ---------------------------------------------------------
# 2. Multi-Head Attention (chapter 3.2.2)
# ---------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    The paper mentions: Multi-head attention allows the model to simultaneously attend to information from different subspaces
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Corresponding to the linear transformation layers of W^Q, W^K, W^V in the paper
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Corresponding to the output projection layer W^O at the end of the paper
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear transformation and split into multiple heads (h=num_heads)
        # Transformed dimension: [Batch, Head, Seq, D_head]
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # 2. Independently calculate Scaled Dot-Product Attention on each head
        x, _ = scaled_dot_product_attention(q, k, v, mask)
        
        # 3. Concatenate the outputs of all heads and project
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_head)
        return self.out_linear(x)


# ---------------------------------------------------------
# 3. Position-wise Feed-Forward (chapter 3.3)
# ---------------------------------------------------------
class FeedForward(nn.Module):
    """
    The paper mentions: After the attention layer, apply the same fully connected network to each position
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# ---------------------------------------------------------
# 4. Positional Encoding (section 3.5 of the paper)
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# ---------------------------------------------------------
# 5. Encoder Layer (corresponding to the left block in Figure 1 of the paper)
# ---------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """
    Core design: Add & Norm (residual connection + layer normalization)
    Note: Here we demonstrate Post-Norm (original structure of the paper), i.e. Norm after the residual connection
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # First part: Attention + residual + Norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_out) # corresponding to Add & Norm in Figure 1 of the paper
        
        # Second part: FFN + residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out) # corresponding to Add & Norm in Figure 1 of the paper
        
        return x

# ---------------------------------------------------------
# 6. Minimum running example
# ---------------------------------------------------------
if __name__ == "__main__":
    # Simulated input parameters
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2

    # Randomly generate input vectors
    x = torch.randn(batch_size, seq_len, d_model)

    # Positional encoding: add PE before encoder (standard Transformer flow)
    pe = PositionalEncoding(d_model)
    x_with_pe = pe(x)
    print(f"Input shape: {x.shape}")
    print(f"After PE shape: {x_with_pe.shape}")
    print(f"PE buffer shape: {pe.pe.shape}")  # [1, max_len, d_model]

    # Instantiate one encoder layer
    layer = TransformerEncoderLayer(d_model, num_heads)

    # Forward propagation (with PE-applied input)
    output = layer(x_with_pe)

    print(f"Output shape: {output.shape}")
    print("Transformer with positional encoding verified!")