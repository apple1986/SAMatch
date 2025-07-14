import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define query, key, and value linear transformations
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Transform input for query, key, and value
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Reshape for matrix multiplication
        B, C, H, W = proj_query.size()
        proj_query = proj_query.view(B, -1, H * W).permute(0, 2, 1)
        proj_key = proj_key.view(B, -1, H * W)

        # Compute attention scores
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Compute weighted sum using attention scores
        proj_value = proj_value.view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return out

# Example usage
input_tensor = torch.randn(24, 16, 128, 128)
self_attention = SelfAttention(in_channels=16)
output = self_attention(input_tensor)
print(output.shape)
