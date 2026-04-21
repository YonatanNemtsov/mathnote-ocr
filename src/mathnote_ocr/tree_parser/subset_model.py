"""Small model that predicts partial tree structure from a subset of symbols.

Given a small set of symbols (3-8) with bounding boxes, predicts for each
symbol: which other symbol is its parent, the edge type, and sibling order.

Architecture:
- Encoder: symbol embeddings + pairwise geometric attention bias (reuses
  the GeoBiasEncoderLayer pattern from parser/model.py)
- Biaffine head: scores each directed pair (parent → child)
- Edge type head: MLP classifies edge type for each predicted edge
- Order head: MLP predicts sibling order
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mathnote_ocr.tree_parser.tree import NUM_EDGE_TYPES, ROOT

# ── Encoder (lightweight geo-bias transformer) ───────────────────────


class GeoBiasEncoderLayer(nn.Module):
    """Encoder layer with learned pairwise geometric attention bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_geo_buckets: int = 8,
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

        self.v_off_bias = nn.Embedding(n_geo_buckets, n_heads)
        self.h_off_bias = nn.Embedding(n_geo_buckets, n_heads)
        self.size_bias = nn.Embedding(n_geo_buckets, n_heads)

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
        x: torch.Tensor,  # (B, S, D)
        geo_buckets: torch.Tensor,  # (B, 3, S, S) long
        key_padding_mask: torch.Tensor | None = None,  # (B, S) True=pad
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        B, S, _ = x.shape

        q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        geo_bias = (
            self.v_off_bias(geo_buckets[:, 0])
            + self.h_off_bias(geo_buckets[:, 1])
            + self.size_bias(geo_buckets[:, 2])
        ).permute(0, 3, 1, 2)

        scores = scores + geo_bias

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.W_o(out)

        x = residual + self.dropout(out)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x


# ── Biaffine layer for parent prediction ─────────────────────────────


class BiaffineScorer(nn.Module):
    """Biaffine attention for scoring directed parent-child pairs.

    For each pair (i, j), computes:
        score(i→j) = h_child_i^T W h_parent_j + U h_child_i + V h_parent_j + b

    Plus a score for the virtual root as parent.
    """

    def __init__(self, d_model: int, d_arc: int) -> None:
        super().__init__()
        # Project to separate child/parent representations
        self.W_child = nn.Linear(d_model, d_arc)
        self.W_parent = nn.Linear(d_model, d_arc)

        # Biaffine weight
        self.U = nn.Parameter(torch.zeros(d_arc, d_arc))
        nn.init.xavier_uniform_(self.U)

        # Bias terms
        self.b_child = nn.Linear(d_arc, 1)
        self.b_parent = nn.Linear(d_arc, 1)

        # Root score: learned vector that acts as "virtual parent"
        self.root_repr = nn.Parameter(torch.randn(d_arc))

    def forward(
        self,
        h: torch.Tensor,  # (B, S, D)
        pad_mask: torch.Tensor | None = None,  # (B, S) True=pad
    ) -> torch.Tensor:
        """Returns parent scores: (B, S, S+1).

        Last column (index S) is the score for the virtual root.
        """
        B, S, _ = h.shape

        h_child = self.W_child(h)  # (B, S, d_arc)
        h_parent = self.W_parent(h)  # (B, S, d_arc)

        # Biaffine: child_i^T U parent_j → (B, S, S)
        # (B, S, d_arc) @ (d_arc, d_arc) → (B, S, d_arc)
        child_U = torch.matmul(h_child, self.U)
        pair_scores = torch.bmm(child_U, h_parent.transpose(1, 2))  # (B, S, S)

        # Add bias terms
        pair_scores = (
            pair_scores
            + self.b_child(h_child)  # (B, S, 1) broadcast over parents
            + self.b_parent(h_parent).transpose(1, 2)  # (B, 1, S) broadcast over children
        )

        # Root scores: each child scored against the root representation
        root_expanded = self.root_repr.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        root_scores = (
            (child_U * root_expanded).sum(dim=-1, keepdim=True)
            + self.b_child(h_child)
            + self.b_parent(root_expanded).sum(dim=-1, keepdim=True)
        )  # (B, S, 1)

        # Concatenate: (B, S, S+1) where last col = root
        scores = torch.cat([pair_scores, root_scores], dim=-1)

        # Mask out padded positions as parents
        if pad_mask is not None:
            # Can't be your own parent
            diag_mask = torch.eye(S, dtype=torch.bool, device=h.device)
            diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)

            # Padded symbols can't be parents
            parent_pad = pad_mask.unsqueeze(1).expand(-1, S, -1)  # (B, S, S)

            # Combined mask (don't include root column)
            combined = diag_mask | parent_pad  # (B, S, S)

            # Add a False column for root (root is always valid)
            root_col = torch.zeros(B, S, 1, dtype=torch.bool, device=h.device)
            full_mask = torch.cat([combined, root_col], dim=-1)  # (B, S, S+1)

            scores = scores.masked_fill(full_mask, float("-inf"))

        return scores


# ── Full subset model ────────────────────────────────────────────────


class SubsetTreeModel(nn.Module):
    """Predicts partial tree structure from a small symbol subset.

    Input: symbol IDs + pairwise geo buckets + per-symbol size features
    Output:
        - parent_scores: (B, S, S+1) — log-probabilities over parents
        - edge_type_scores: (B, S, S+1, E) — edge type logits per parent
        - order_scores: (B, S, S+1) — predicted sibling order per parent
    """

    def __init__(
        self,
        num_symbols: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        d_arc: int = 64,
        n_geo_buckets: int = 8,
        max_symbols: int = 12,
        dropout: float = 0.1,
        num_edge_types: int = NUM_EDGE_TYPES,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_symbols = max_symbols
        self.num_edge_types = num_edge_types

        # Symbol + size embeddings
        self.symbol_embed = nn.Embedding(num_symbols, d_model, padding_idx=0)
        self.height_embed = nn.Embedding(4, d_model)
        self.yoff_embed = nn.Embedding(4, d_model)

        # Encoder layers
        self.enc_layers = nn.ModuleList(
            [
                GeoBiasEncoderLayer(d_model, n_heads, d_ff, n_geo_buckets, dropout)
                for _ in range(n_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # Parent prediction (biaffine)
        self.arc_scorer = BiaffineScorer(d_model, d_arc)

        # Edge type classification
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_edge_types),
        )

        # Edge type for root parent (no parent representation to pair with)
        self.root_edge_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_edge_types),
        )

        # Previous sibling (SEQ) prediction
        self.seq_scorer = BiaffineScorer(d_model, d_arc)

        # Sibling order prediction
        self.order_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.root_order_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.symbol_embed.weight, 0, 0.02)

    def encode(
        self,
        symbol_ids: torch.Tensor,  # (B, S)
        geo_buckets: torch.Tensor,  # (B, 3, S, S) long
        pad_mask: torch.Tensor,  # (B, S) True=pad
        size_feats: torch.Tensor | None = None,  # (B, S, 2) long
    ) -> torch.Tensor:
        """Encode symbols into contextual representations."""
        x = self.symbol_embed(symbol_ids)
        if size_feats is not None:
            x = x + self.height_embed(size_feats[:, :, 0])
            x = x + self.yoff_embed(size_feats[:, :, 1])

        for layer in self.enc_layers:
            x = layer(x, geo_buckets, pad_mask)

        return self.enc_norm(x)

    def forward(
        self,
        symbol_ids: torch.Tensor,  # (B, S)
        geo_buckets: torch.Tensor,  # (B, 3, S, S)
        pad_mask: torch.Tensor,  # (B, S)
        size_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Returns dict with:
            parent_scores: (B, S, S+1) — logits for parent selection
            edge_type_scores: (B, S, S+1, E) — logits for edge type
            order_preds: (B, S, S+1) — predicted order values (regression)
        """
        _B, S = symbol_ids.shape
        h = self.encode(symbol_ids, geo_buckets, pad_mask, size_feats)

        # Parent scores
        parent_scores = self.arc_scorer(h, pad_mask)  # (B, S, S+1)

        # SEQ (previous sibling) scores
        seq_scores = self.seq_scorer(h, pad_mask)  # (B, S, S+1)

        # Edge type scores for each (child, possible_parent) pair
        # Build pairwise features: concat child repr with each parent repr
        h_child = h.unsqueeze(2).expand(-1, -1, S, -1)  # (B, S, S, D)
        h_parent = h.unsqueeze(1).expand(-1, S, -1, -1)  # (B, S, S, D)
        pair_feats = torch.cat([h_child, h_parent], dim=-1)  # (B, S, S, 2D)

        edge_sym = self.edge_type_mlp(pair_feats)  # (B, S, S, E)
        edge_root = self.root_edge_mlp(h)  # (B, S, E)

        edge_type_scores = torch.cat(
            [edge_sym, edge_root.unsqueeze(2)],
            dim=2,
        )  # (B, S, S+1, E)

        # Order predictions
        order_sym = self.order_mlp(pair_feats).squeeze(-1)  # (B, S, S)
        order_root = self.root_order_mlp(h).squeeze(-1)  # (B, S)

        order_preds = torch.cat(
            [order_sym, order_root.unsqueeze(2)],
            dim=2,
        )  # (B, S, S+1)

        return {
            "parent_scores": parent_scores,
            "edge_type_scores": edge_type_scores,
            "order_preds": order_preds,
            "seq_scores": seq_scores,
        }

    def predict(
        self,
        symbol_ids: torch.Tensor,
        geo_buckets: torch.Tensor,
        pad_mask: torch.Tensor,
        size_feats: torch.Tensor | None = None,
        n_real: int | None = None,
    ) -> list[tuple[int, int, float, int]]:
        """Predict tree for a single example (no batch dim).

        Returns list of (parent_idx, edge_type, order, seq_prev) per symbol.
        parent_idx = ROOT (-1) for root-level symbols.
        seq_prev = ROOT (-1) for first children (no previous sibling).
        """
        # Add batch dimension
        out = self.forward(
            symbol_ids.unsqueeze(0),
            geo_buckets.unsqueeze(0),
            pad_mask.unsqueeze(0),
            size_feats.unsqueeze(0) if size_feats is not None else None,
        )

        S = symbol_ids.shape[0]
        if n_real is None:
            n_real = (~pad_mask).sum().item()

        parent_scores = out["parent_scores"][0]  # (S, S+1)
        edge_type_scores = out["edge_type_scores"][0]  # (S, S+1, E)
        order_preds = out["order_preds"][0]  # (S, S+1)
        seq_scores = out["seq_scores"][0]  # (S, S+1)

        results = []
        for i in range(n_real):
            # Best parent
            parent_idx = parent_scores[i].argmax().item()
            if parent_idx == S:  # root
                parent_idx = ROOT

            # Edge type for this parent
            if parent_idx == ROOT:
                et = edge_type_scores[i, S].argmax().item()
                order = order_preds[i, S].item()
            else:
                et = edge_type_scores[i, parent_idx].argmax().item()
                order = order_preds[i, parent_idx].item()

            # Previous sibling
            seq_idx = seq_scores[i].argmax().item()
            seq_prev = ROOT if seq_idx == S else seq_idx

            results.append((parent_idx, et, order, seq_prev))

        return results


# ── Factory ──────────────────────────────────────────────────────────


def load_subset_model(ckpt: dict, device: torch.device = None):
    """Create model from a checkpoint."""
    cfg = {k: v for k, v in ckpt["config"].items() if k != "model_version"}
    # Infer num_edge_types from checkpoint weights if not in config
    if "num_edge_types" not in cfg and "edge_type_mlp.3.weight" in ckpt["model_state_dict"]:
        cfg["num_edge_types"] = ckpt["model_state_dict"]["edge_type_mlp.3.weight"].shape[0]
    model = SubsetTreeModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
