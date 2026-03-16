import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class ViTTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm)
        x = x + self.ff(self.norm2(x))
        return x