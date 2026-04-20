"""Loss computation for SubsetTreeModel training."""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_loss(
    out: dict[str, torch.Tensor],
    parent_targets: torch.Tensor,    # (B, S)
    edge_targets: torch.Tensor,      # (B, S)
    order_targets: torch.Tensor,     # (B, S)
    pad_mask: torch.Tensor,          # (B, S)
    seq_targets: torch.Tensor | None = None,  # (B, S)
    order_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined loss and metrics.

    Returns (loss, metrics_dict).
    """
    B, S = parent_targets.shape
    device = parent_targets.device

    # ── Parent loss: CE over S+1 classes ──
    # parent_scores: (B, S, S+1)
    parent_scores = out["parent_scores"]
    parent_loss = nn.functional.cross_entropy(
        parent_scores.view(-1, S + 1),
        parent_targets.view(-1),
        ignore_index=-100,
    )

    # ── Valid masks ──
    # All non-padded symbols with known parent (for parent acc + order loss)
    valid_all = (~pad_mask) & (parent_targets != -100)  # (B, S)
    # Only structural children (for edge loss — ROOT children have edge=-1)
    valid_edge = valid_all & (edge_targets != -100)  # (B, S)

    edge_type_scores = out["edge_type_scores"]  # (B, S, S+1, E)
    order_preds_out = out["order_preds"]  # (B, S, S+1)

    # ── Edge type loss: CE only for structural children ──
    if valid_edge.any():
        eidx = valid_edge.nonzero(as_tuple=False)  # (Ve, 2)
        eb, es = eidx[:, 0], eidx[:, 1]
        ep = parent_targets[eb, es]

        edge_logits = edge_type_scores[eb, es, ep]  # (Ve, E)
        edge_tgt = edge_targets[eb, es]  # (Ve,)
        edge_loss = nn.functional.cross_entropy(edge_logits, edge_tgt)

        with torch.no_grad():
            edge_pred = edge_logits.argmax(dim=-1)
            edge_acc = (edge_pred == edge_tgt).float().mean().item()
    else:
        edge_loss = torch.tensor(0.0, device=device)
        edge_acc = 0.0

    # ── Order loss + parent acc: all valid symbols (including ROOT children) ──
    if valid_all.any():
        aidx = valid_all.nonzero(as_tuple=False)  # (Va, 2)
        ab, a_s = aidx[:, 0], aidx[:, 1]
        ap = parent_targets[ab, a_s]

        pred_order = order_preds_out[ab, a_s, ap]  # (Va,)
        true_order = order_targets[ab, a_s]  # (Va,)
        order_loss = nn.functional.l1_loss(pred_order, true_order)

        with torch.no_grad():
            parent_pred = parent_scores[ab, a_s].argmax(dim=-1)
            parent_acc = (parent_pred == ap).float().mean().item()
            order_mae = (pred_order - true_order).abs().mean().item()
    else:
        order_loss = torch.tensor(0.0, device=device)
        parent_acc = 0.0
        order_mae = 0.0

    # ── SEQ loss: CE over S+1 classes (previous sibling prediction) ──
    seq_loss = torch.tensor(0.0, device=device)
    seq_acc = 0.0
    if seq_targets is not None and "seq_scores" in out:
        seq_scores = out["seq_scores"]  # (B, S, S+1)
        valid_seq = (~pad_mask) & (seq_targets != -100)
        if valid_seq.any():
            seq_loss = nn.functional.cross_entropy(
                seq_scores.view(-1, S + 1),
                seq_targets.view(-1),
                ignore_index=-100,
            )
            with torch.no_grad():
                sidx = valid_seq.nonzero(as_tuple=False)
                sb, ss = sidx[:, 0], sidx[:, 1]
                seq_pred = seq_scores[sb, ss].argmax(dim=-1)
                seq_tgt = seq_targets[sb, ss]
                seq_acc = (seq_pred == seq_tgt).float().mean().item()

    loss = parent_loss + edge_loss + seq_loss + order_weight * order_loss

    metrics = {
        "parent_loss": parent_loss.item(),
        "edge_loss": edge_loss.item(),
        "order_loss": order_loss.item(),
        "seq_loss": seq_loss.item(),
        "parent_acc": parent_acc,
        "edge_acc": edge_acc,
        "seq_acc": seq_acc,
        "order_mae": order_mae,
    }

    return loss, metrics
