import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v):
        scale = self.d_head
        scores = q @ k.transpose(-2, -1) / math.sqrt(scale)
        weights = torch.softmax(scores, dim=-1) 

        output = weights @ v
        return output
    
    def forward(self, q, k, v):
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_head)
        batch_size = q.size(0)
        Q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.wk(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.wv(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        attention_score:torch.Tensor = self.scaled_dot_product_attention(Q, K, V)

        # (batch_size, num_heads, seq_len, d_head) -> (batch_size, seq_len, d_model)
        combined_attention_score = attention_score.transpose(2, 1).contiguous().view(batch_size, -1, self.d_model)

        output = self.fc(combined_attention_score)
        return output


if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    batch_size = 64
    seq_len = 10

    attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    input_tensor = torch.rand((batch_size, seq_len, d_model))

    output_tensor = attention(input_tensor, input_tensor, input_tensor)

    print(input_tensor.shape)
    print(output_tensor.shape)
