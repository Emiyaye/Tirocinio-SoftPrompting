from typing import List, Dict, Optional

import torch
import torch.nn as nn


class _PromptEncoderBlock(nn.Module):
    """Pre-LN transformer decoder block: self-attn → cross-attn → FFN.

    Operates on a batch of labels in parallel. Shapes:
      - ``x``                : (N, K, D) queries.
      - ``kv``               : (N, T_max, D) padded description tokens.
      - ``kv_key_padding``   : (N, T_max) with True at padded positions (the
                               convention expected by ``nn.MultiheadAttention``).
    """

    def __init__(
        self,
        hidden_size: int,
        attention_heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        D = hidden_size

        self.norm_self = nn.LayerNorm(D)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=attention_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm_cross = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=attention_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm_ffn = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, ffn_mult * D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * D, D),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention over the K queries (each label independently)
        h = self.norm_self(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention from queries onto each label's description tokens
        h = self.norm_cross(x)
        h, _ = self.cross_attn(h, kv, kv, key_padding_mask=kv_key_padding)
        x = x + h

        # Feed-forward
        h = self.norm_ffn(x)
        h = self.ffn(h)
        x = x + h

        return x
