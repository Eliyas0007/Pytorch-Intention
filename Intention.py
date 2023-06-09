import torch
import torch.nn as nn
from einops import rearrange

class Intention(nn.Module):

    def __init__(self, embed_dim, num_head, device) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.device = device
        self.alpha = nn.Parameter(torch.rand(1))
        assert embed_dim % num_head == 0, 'embed_dim must be divisible by num_head'

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.soft_max = nn.Softmax(dim=-1)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, decoder_in: torch.Tensor = None, mask=False):

        value = self.value(x)
        key = self.key(x)
        if decoder_in is not None:
            query = self.query(decoder_in)
        else:
            query = self.query(x)

        query = rearrange(query, 'b s (nh hd) -> b nh s hd', nh=self.num_head, hd=self.head_dim)
        key_T = rearrange(key.clone(), 'b s (nh hd) -> b nh hd s', nh=self.num_head, hd=self.head_dim)
        key = rearrange(key, 'b s (nh hd) -> b nh s hd', nh=self.num_head, hd=self.head_dim)
        value = rearrange(value, 'b s (nh hd) -> b nh s hd', nh=self.num_head, hd=self.head_dim)

        kk = key_T @ key
        kk = self.alpha * torch.eye(kk.shape[-1], device=self.device) + kk
        kk_inv = torch.inverse(kk)
        attention_score = (kk_inv @ key_T) @ value

        if mask:
            size = attention_score.shape[-1]
            attention_score = attention_score + torch.triu(torch.full((size, size), float('-inf'), device=self.device, requires_grad=False), diagonal=1)

        attention_score_scaled = self.soft_max(attention_score)

        out = query @ attention_score_scaled
        out = rearrange(out, 'b nh s hd -> b s (nh hd)')
        out = self.fc(out)

        return out

