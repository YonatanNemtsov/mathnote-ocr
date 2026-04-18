"""EvidenceGNN model — refines aggregated evidence into final tree scores.

Takes aggregated evidence from subset model predictions and refines it
using a small transformer with evidence-biased attention. Replaces the
heuristic propagation step with learned message passing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tree_parser.tree import NUM_EDGE_TYPES
from tree_parser.subset_model import BiaffineScorer


class EvidenceBiasLayer(nn.Module):
    """Transformer layer with attention biased by evidence edge features.

    Like GeoBiasEncoderLayer but takes continuous edge features (d_edge dims)
    instead of discrete geo_buckets. Projects edge features to per-head
    attention biases via a linear layer.
    """

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

        # Continuous edge features → per-head attention bias
        self.edge_bias = nn.Linear(d_edge, n_heads)

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
        x: torch.Tensor,                # (B, N, D)
        edge_feats: torch.Tensor,        # (B, N, N, d_edge)
        key_padding_mask: torch.Tensor | None = None,  # (B, N) True=pad
        adj_mask: torch.Tensor | None = None,  # (B, N, N) True=connected
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        B, N, _ = x.shape

        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add evidence-based attention bias
        bias = self.edge_bias(edge_feats)     # (B, N, N, n_heads)
        bias = bias.permute(0, 3, 1, 2)       # (B, n_heads, N, N)
        scores = scores + bias

        # Block attention to non-connected nodes
        if adj_mask is not None:
            scores = scores.masked_fill(
                ~adj_mask.unsqueeze(1), float("-inf")
            )

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)

        x = residual + self.dropout(out)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x


class EvidenceGNN(nn.Module):
    """Refines aggregated evidence into final tree scores.

    Input:
        symbol_ids: (B, N) — symbol vocabulary IDs
        size_feats: (B, N, 2) — height bucket + y-offset bucket
        edge_features: (B, N, N+1, d_edge) — from evidence_to_features()
        pad_mask: (B, N) — True for padding positions

    Output:
        parent_scores: (B, N, N+1) — logits for parent selection
        edge_type_scores: (B, N, N+1, E) — logits for edge type
        seq_scores: (B, N, N+1) — logits for previous sibling selection
    """

    def __init__(
        self,
        num_symbols: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        d_edge: int = 11,
        d_arc: int = 32,
        dropout: float = 0.1,
        sparse_attention: bool = False,
        size_mode: str = "bucketed",  # "bucketed" or "continuous"
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sparse_attention = sparse_attention
        self.size_mode = size_mode

        # Node encoding: symbol identity + spatial features
        self.symbol_embed = nn.Embedding(num_symbols, d_model, padding_idx=0)
        if size_mode == "continuous":
            self.size_proj = nn.Linear(2, d_model)
        else:
            self.height_embed = nn.Embedding(4, d_model)
            self.yoff_embed = nn.Embedding(4, d_model)

        # Project ROOT evidence column → node feature
        # edge_features[:, :, N, :] tells each node how "rooty" it is
        self.root_evidence_proj = nn.Linear(d_edge, d_model)

        # Transformer layers with evidence-biased attention
        self.layers = nn.ModuleList([
            EvidenceBiasLayer(d_model, n_heads, d_ff, d_edge, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output: parent prediction (biaffine scorer)
        self.parent_scorer = BiaffineScorer(d_model, d_arc)

        # Output: previous sibling prediction (biaffine scorer)
        self.seq_scorer = BiaffineScorer(d_model, d_arc)

        # Output: edge type classification
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_EDGE_TYPES),
        )
        self.root_edge_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_EDGE_TYPES),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.symbol_embed.weight, 0, 0.02)

    def forward(
        self,
        symbol_ids: torch.Tensor,      # (B, N)
        size_feats: torch.Tensor,       # (B, N, 2)
        edge_features: torch.Tensor,    # (B, N, N+1, d_edge)
        pad_mask: torch.Tensor,         # (B, N) True=pad
    ) -> dict[str, torch.Tensor]:
        B, N = symbol_ids.shape

        # Node encoding
        h = self.symbol_embed(symbol_ids)
        if self.size_mode == "continuous":
            h = h + self.size_proj(size_feats.float())
        else:
            h = h + self.height_embed(size_feats[:, :, 0])
            h = h + self.yoff_embed(size_feats[:, :, 1])

        # Add ROOT evidence as node feature
        root_ev = edge_features[:, :, N, :]           # (B, N, d_edge)
        h = h + self.root_evidence_proj(root_ev)

        # Edge features between real nodes (N×N) for attention bias
        edge_nn = edge_features[:, :, :N, :]          # (B, N, N, d_edge)

        # Adjacency mask: only attend to nodes with evidence
        adj_mask = None
        if self.sparse_attention:
            # vote_density is at index NUM_EDGE_TYPES (after E vote fractions)
            vote_density = edge_nn[:, :, :, NUM_EDGE_TYPES]  # (B, N, N)
            # Connected if either direction has votes
            adj_mask = (vote_density > 0) | (vote_density.transpose(1, 2) > 0)
            # Always connect self
            adj_mask = adj_mask | torch.eye(N, dtype=torch.bool, device=adj_mask.device)
            # Don't mask padded positions (key_padding_mask handles those)
            adj_mask = adj_mask | pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, edge_nn, pad_mask, adj_mask)
        h = self.norm(h)

        # Parent scores (biaffine)
        parent_scores = self.parent_scorer(h, pad_mask)  # (B, N, N+1)

        # SEQ scores (biaffine) — previous sibling prediction
        seq_scores = self.seq_scorer(h, pad_mask)        # (B, N, N+1)

        # Edge type scores
        h_child = h.unsqueeze(2).expand(-1, -1, N, -1)     # (B, N, N, D)
        h_parent = h.unsqueeze(1).expand(-1, N, -1, -1)    # (B, N, N, D)
        pair_feats = torch.cat([h_child, h_parent], dim=-1) # (B, N, N, 2D)

        edge_sym = self.edge_type_mlp(pair_feats)           # (B, N, N, E)
        edge_root = self.root_edge_mlp(h)                    # (B, N, E)

        edge_type_scores = torch.cat(
            [edge_sym, edge_root.unsqueeze(2)], dim=2,
        )  # (B, N, N+1, E)

        return {
            "parent_scores": parent_scores,
            "edge_type_scores": edge_type_scores,
            "seq_scores": seq_scores,
        }
