"""StrokeGNN: graph neural network for stroke grouping and classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Edge-biased transformer layer ────────────────────────────────────


class EdgeBiasTransformerLayer(nn.Module):
    """Transformer layer with continuous edge features as attention bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_edge: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Project edge features → per-head bias
        self.edge_proj = nn.Linear(d_edge, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, D)
        edge_feats: torch.Tensor,      # (B, N, N, d_edge)
        pad_mask: torch.Tensor | None = None,  # (B, N) True=pad
        adj_mask: torch.Tensor | None = None,  # (B, N, N) True=connected
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        B, N, _ = x.shape

        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Edge bias: (B, N, N, d_edge) → (B, N, N, n_heads) → (B, n_heads, N, N)
        edge_bias = self.edge_proj(edge_feats).permute(0, 3, 1, 2)
        scores = scores + edge_bias

        # Adjacency mask: block attention to non-neighbours
        if adj_mask is not None:
            scores = scores.masked_fill(
                ~adj_mask.unsqueeze(1), float("-inf")
            )

        if pad_mask is not None:
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)  # all-masked rows → 0 instead of NaN
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)

        x = residual + self.dropout(out)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x


# ── Biaffine scorer for edge prediction ──────────────────────────────


class SymmetricBiaffineScorer(nn.Module):
    """Symmetric biaffine scorer for undirected edge prediction.

    score(i, j) = h_i^T W h_j + b_i + b_j
    Symmetrized: (score(i,j) + score(j,i)) / 2
    """

    def __init__(self, d_model: int, d_arc: int) -> None:
        super().__init__()
        self.W_left = nn.Linear(d_model, d_arc)
        self.W_right = nn.Linear(d_model, d_arc)
        self.U = nn.Parameter(torch.zeros(d_arc, d_arc))
        nn.init.xavier_uniform_(self.U)
        self.bias = nn.Linear(d_arc, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns symmetric edge scores: (B, N, N)."""
        h_l = self.W_left(h)   # (B, N, d_arc)
        h_r = self.W_right(h)  # (B, N, d_arc)

        # Biaffine: h_l^T U h_r
        lU = torch.matmul(h_l, self.U)  # (B, N, d_arc)
        scores = torch.bmm(lU, h_r.transpose(1, 2))  # (B, N, N)
        scores = scores + self.bias(h_l)  # (B, N, 1) broadcast
        scores = scores + self.bias(h_r).transpose(1, 2)  # (B, 1, N) broadcast

        # Symmetrize
        scores = (scores + scores.transpose(1, 2)) / 2
        return scores


# ── Main model ───────────────────────────────────────────────────────


class StrokeGNN(nn.Module):
    """GNN for stroke grouping: predicts same-symbol edges + per-stroke labels.

    Node features:
        - 32×32 mini-render per stroke → CNN → 32-dim
        - 8 geometric features → linear → 16-dim
        - Total: 48-dim

    Architecture:
        - 3 transformer layers with edge-biased attention
        - Symmetric biaffine scorer for edge prediction
        - Linear head for node (symbol) classification
    """

    def __init__(
        self,
        num_classes: int,
        render_size: int = 32,
        d_render: int = 32,
        d_geo: int = 16,
        n_geo_feats: int = 8,
        d_edge: int = 5,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.render_size = render_size
        d_node = d_render + d_geo

        # Stroke mini-render CNN encoder
        # 32→16→8 after two pool layers, flatten = 32*8*8 = 2048
        cnn_flat = 32 * (render_size // 4) ** 2
        self.stroke_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(cnn_flat, d_render),
        )

        # Geometric feature projection
        self.geo_proj = nn.Linear(n_geo_feats, d_geo)

        # Transformer layers
        d_ff = d_node * 2
        self.layers = nn.ModuleList([
            EdgeBiasTransformerLayer(d_node, n_heads, d_ff, d_edge, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_node)

        # Edge classifier (same_symbol / different_symbol)
        self.edge_scorer = SymmetricBiaffineScorer(d_node, d_arc=32)

        # Node classifier (symbol label)
        self.node_classifier = nn.Linear(d_node, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        renders: torch.Tensor,       # (B, N, 1, R, R)
        geo: torch.Tensor,           # (B, N, 8)
        edge_feats: torch.Tensor,    # (B, N, N, 5)
        pad_mask: torch.Tensor | None = None,  # (B, N) True=pad
        adj_mask: torch.Tensor | None = None,  # (B, N, N) True=connected
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            edge_scores: (B, N, N) — logits for same_symbol (> 0 = same)
            node_logits: (B, N, num_classes) — symbol classification logits
        """
        B, N = renders.shape[:2]
        R = self.render_size

        # Encode renders: (B*N, 1, R, R) → (B*N, d_render) → (B, N, d_render)
        cnn_in = renders.reshape(B * N, 1, R, R)
        cnn_out = self.stroke_cnn(cnn_in).reshape(B, N, -1)

        # Encode geometry
        geo_out = self.geo_proj(geo)  # (B, N, d_geo)

        # Combine
        x = torch.cat([cnn_out, geo_out], dim=-1)  # (B, N, d_node)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, edge_feats, pad_mask, adj_mask)

        x = self.final_norm(x)

        # Predictions
        edge_scores = self.edge_scorer(x)          # (B, N, N)
        node_logits = self.node_classifier(x)       # (B, N, num_classes)

        return edge_scores, node_logits
