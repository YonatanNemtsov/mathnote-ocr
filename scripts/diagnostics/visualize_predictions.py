"""Visualize tree predictions vs ground truth.

Usage:
    python3.10 tools/visualize_predictions.py --run v3
    python3.10 tools/visualize_predictions.py --run v3 --exhaustive
"""

import json
import random
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree import (
    SymbolNode, build_tree, tree_to_latex, ROOT,
    NUM, DEN, SUP, SUB, SQRT_CONTENT, EDGE_NAMES, NUM_EDGE_TYPES,
)
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft, evidence_to_features
from mathnote_ocr.tree_parser.propagation import propagate_seq
from mathnote_ocr.tree_parser.subset_selection import make_spatial_subsets, _bbox_edge_dist
from mathnote_ocr.tree_parser.tree_builder import build_tree_from_evidence, build_tree_from_scores, find_seq_conflicts

EDGE_COLORS = {
    -1: "#666666",
    NUM: "#2196F3",
    DEN: "#F44336",
    SUP: "#4CAF50",
    SUB: "#FF9800",
    SQRT_CONTENT: "#9C27B0",
}


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _dot_id(node: SymbolNode, prefix: str = "n") -> str:
    return f"{prefix}{node.index}"


def _build_dot(roots: list[SymbolNode], glyph_names: list[str], prefix: str = "n") -> str:
    lines = [
        "digraph T {",
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fillcolor="#f5f5f5",'
        '        fontname="Courier", fontsize=14, margin="0.15,0.07"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    all_nodes: list[SymbolNode] = []

    def _collect(node: SymbolNode):
        all_nodes.append(node)
        for et, children in sorted(node.children.items()):
            for c in children:
                _collect(c)

    for r in roots:
        _collect(r)

    for node in all_nodes:
        label = _escape(glyph_names[node.index])
        lines.append(f'  {_dot_id(node, prefix)} [label="{label}"];')

    # SEQ arrows between root-level siblings
    sorted_roots = sorted(roots, key=lambda r: r.order)
    for i in range(1, len(sorted_roots)):
        prev_r = sorted_roots[i - 1]
        curr_r = sorted_roots[i]
        lines.append(
            f'  {_dot_id(prev_r, prefix)} -> {_dot_id(curr_r, prefix)}'
            f' [style=dashed, color="#AAAAAA", constraint=false,'
            f'  arrowsize=0.6, label=" seq", fontsize=8,'
            f'  fontcolor="#AAAAAA"];'
        )

    for node in all_nodes:
        for et, children in sorted(node.children.items()):
            color = EDGE_COLORS.get(et, "#000000")
            ename = EDGE_NAMES[et]
            sorted_children = sorted(children, key=lambda c: c.order)
            for c in sorted_children:
                lines.append(
                    f'  {_dot_id(node, prefix)} -> {_dot_id(c, prefix)}'
                    f' [label=" {ename}", color="{color}",'
                    f'  fontcolor="{color}"];'
                )
            # SEQ arrows: dashed edges between consecutive siblings
            for i in range(1, len(sorted_children)):
                prev_c = sorted_children[i - 1]
                curr_c = sorted_children[i]
                lines.append(
                    f'  {_dot_id(prev_c, prefix)} -> {_dot_id(curr_c, prefix)}'
                    f' [style=dashed, color="#AAAAAA", constraint=false,'
                    f'  arrowsize=0.6, label=" seq", fontsize=8,'
                    f'  fontcolor="#AAAAAA"];'
                )

    lines.append("}")
    return "\n".join(lines)


def _dot_to_svg(dot_src: str) -> str:
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_src, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"<pre>graphviz error: {_escape(result.stderr)}</pre>"
    svg = result.stdout
    idx = svg.find("<svg")
    return svg[idx:] if idx >= 0 else svg


def _build_trees(symbols, tree_labels) -> tuple[list[SymbolNode], list[str]]:
    """Build tree from symbols + tree labels."""
    names = [s["name"] for s in symbols]
    nodes = []
    for j, (s, t) in enumerate(zip(symbols, tree_labels)):
        nodes.append(SymbolNode(
            symbol=s["name"],
            bbox=s["bbox"],
            index=j,
            parent=t["parent"],
            edge_type=t["edge_type"],
            order=t["order"],
        ))
    roots = build_tree(nodes)
    return roots, names


@torch.no_grad()
def predict_tree_exhaustive(subset_model, symbol_vocab, symbols, device,
                            max_subset=8, radius_mult=4.0):
    """Deterministic: for each symbol, take all neighbors within a radius.

    Uses bbox edge-to-edge distance via _spatial_subsets.
    Subsets are clamped to [3, max_subset].
    No conflict resolution — just one pass of spatial subsets.
    """
    N = len(symbols)
    bbox_lists = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    subsets = _spatial_subsets(symbols, max_subset, radius_mult)
    all_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset)

    evidence = aggregate_evidence_soft(N, all_partial)
    propagate_seq(evidence)
    roots = build_tree_from_evidence(evidence, names, bbox_lists)
    return roots, names


def _spatial_subsets(symbols, max_subset, radius_mult):
    """Generate one subset per symbol using spatial radius."""
    bboxes = [s["bbox"] for s in symbols]
    return make_spatial_subsets(bboxes, max_subset, radius_mult)


def _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset):
    """Run subset model on a list of subsets, return raw outputs for soft aggregation."""
    partial_outputs = []
    for subset_indices in subsets:
        n_sub = len(subset_indices)
        sub_S = max_subset

        sub_ids = torch.zeros(sub_S, dtype=torch.long, device=device)
        for i, gi in enumerate(subset_indices):
            name = symbols[gi]["name"]
            sub_ids[i] = symbol_vocab.get(name, symbol_vocab.get("<unk>", 1))

        bbox_list = [symbols[gi]["bbox"] for gi in subset_indices]
        geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, sub_S)
        geo_buckets = geo_buckets.to(device)
        size_feats = size_feats.to(device)

        sub_pad = torch.ones(sub_S, dtype=torch.bool, device=device)
        sub_pad[:n_sub] = False

        out = subset_model.forward(
            sub_ids.unsqueeze(0), geo_buckets.unsqueeze(0),
            sub_pad.unsqueeze(0), size_feats.unsqueeze(0),
        )
        out_cpu = {k: v[0].cpu() for k, v in out.items()}
        partial_outputs.append((subset_indices, out_cpu, n_sub))
    return partial_outputs


@torch.no_grad()
def predict_tree_iterative(subset_model, symbol_vocab, symbols, device,
                           max_subset=8, radius_mult=4.0, max_iters=3):
    """Build tree using iterative SEQ-conflict targeted subset sampling.

    1. Spatial subsets → evidence → Edmonds tree
    2. Find SEQ conflicts → targeted subsets around conflicts
    3. Re-aggregate → rebuild → repeat
    """
    N = len(symbols)
    bbox_lists = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    # Initial subsets (spatial)
    subsets = _spatial_subsets(symbols, max_subset, radius_mult)
    all_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset)

    for _ in range(max_iters):
        evidence = aggregate_evidence_soft(N, all_partial)
        propagate_seq(evidence)
        roots = build_tree_from_evidence(evidence, names, bbox_lists)
        targets = find_seq_conflicts(
            evidence, roots, bbox_lists,
            seq_threshold=2.0, max_subset_size=min(max_subset, N),
        )
        if not targets:
            break
        new_trees = _run_subsets(subset_model, symbol_vocab, symbols, device, targets, max_subset)
        all_partial.extend(new_trees)

    evidence = aggregate_evidence_soft(N, all_partial)
    propagate_seq(evidence)
    roots = build_tree_from_evidence(evidence, names, bbox_lists)
    return roots, names


@torch.no_grad()
def predict_tree_gnn(gnn_model, subset_model, symbol_vocab, symbols, device,
                     max_subset=8, radius_mult=4.0):
    """Build tree using GNN refinement of evidence.

    subsets → evidence → GNN → Edmonds → tree
    """
    N = len(symbols)
    bbox_lists = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    # Get evidence from subset model
    subsets = _spatial_subsets(symbols, max_subset, radius_mult)
    all_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset)
    evidence = aggregate_evidence_soft(N, all_partial)
    _, edge_features = evidence_to_features(evidence)

    # Prepare GNN inputs
    unk_id = symbol_vocab.get("<unk>", 1)
    sym_ids = torch.tensor(
        [symbol_vocab.get(names[i], unk_id) for i in range(N)],
        dtype=torch.long, device=device,
    )
    _, size_feats = compute_features_from_bbox_list(bbox_lists, N)
    size_feats = size_feats.to(device)
    edge_features = edge_features.to(device)
    pad_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Run GNN
    out = gnn_model(
        sym_ids.unsqueeze(0),
        size_feats.unsqueeze(0),
        edge_features.unsqueeze(0),
        pad_mask.unsqueeze(0),
    )

    parent_scores = out["parent_scores"][0]          # (N, N+1)
    edge_type_scores = out["edge_type_scores"][0]    # (N, N+1, E)
    order_preds = torch.zeros(N, N + 1)              # unused, spatial sort handles order

    roots = build_tree_from_scores(
        parent_scores, edge_type_scores, order_preds,
        names, bbox_lists,
    )
    return roots, names


def _run_gnn(gnn_model, evidence, names, bboxes, symbol_vocab, device):
    """Run GNN on evidence, return tree roots."""
    N = len(names)
    _, edge_features = evidence_to_features(evidence)

    unk_id = symbol_vocab.get("<unk>", 1)
    sym_ids = torch.tensor(
        [symbol_vocab.get(names[i], unk_id) for i in range(N)],
        dtype=torch.long, device=device,
    )
    _, size_feats = compute_features_from_bbox_list(bboxes, N)
    size_feats = size_feats.to(device)
    edge_features = edge_features.to(device)
    pad_mask = torch.zeros(N, dtype=torch.bool, device=device)

    out = gnn_model(
        sym_ids.unsqueeze(0),
        size_feats.unsqueeze(0),
        edge_features.unsqueeze(0),
        pad_mask.unsqueeze(0),
    )

    parent_scores = out["parent_scores"][0]
    edge_type_scores = out["edge_type_scores"][0]
    order_preds = torch.zeros(N, N + 1)

    return build_tree_from_scores(
        parent_scores, edge_type_scores, order_preds,
        names, bboxes,
    )


@torch.no_grad()
def predict_tree_gnn_iterative(gnn_model, subset_model, symbol_vocab, symbols, device,
                                max_subset=8, radius_mult=4.0, max_iters=3):
    """GNN + iterative SEQ-conflict resolution.

    1. Spatial subsets → evidence → GNN → tree
    2. Find SEQ conflicts → targeted subsets around conflicts
    3. Re-aggregate evidence → GNN → rebuild → repeat
    """
    N = len(symbols)
    bbox_lists = [s["bbox"] for s in symbols]
    names = [s["name"] for s in symbols]

    subsets = _spatial_subsets(symbols, max_subset, radius_mult)
    all_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, subsets, max_subset)

    for _ in range(max_iters):
        evidence = aggregate_evidence_soft(N, all_partial)
        roots = _run_gnn(gnn_model, evidence, names, bbox_lists, symbol_vocab, device)
        targets = find_seq_conflicts(
            evidence, roots, bbox_lists,
            seq_threshold=2.0, max_subset_size=min(max_subset, N),
        )
        if not targets:
            break
        new_partial = _run_subsets(subset_model, symbol_vocab, symbols, device, targets, max_subset)
        all_partial.extend(new_partial)

    evidence = aggregate_evidence_soft(N, all_partial)
    roots = _run_gnn(gnn_model, evidence, names, bbox_lists, symbol_vocab, device)
    return roots, names


def generate_html(val_path, run_name, out_path, n_examples=15, seed=42,
                  filter_latex=None, exhaustive=False):
    random.seed(seed)

    # Load subset model
    device = torch.device("cpu")
    ckpt = load_checkpoint("tree_subset", run_name, device=device)
    cfg = ckpt["config"]
    symbol_vocab = ckpt["symbol_vocab"]
    max_subset = cfg.get("max_symbols", 8)

    model = SubsetTreeModel(
        num_symbols=cfg["num_symbols"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        d_arc=cfg["d_arc"],
        max_symbols=max_subset,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_symbols = 20 if exhaustive else max_subset

    # Load validation examples
    examples = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n = len(ex["symbols"])
            if n < 2 or n > max_symbols:
                continue
            if filter_latex and filter_latex not in ex["latex"]:
                continue
            examples.append(ex)

    random.shuffle(examples)
    examples = examples[:n_examples]

    # Generate cards
    cards = []
    n_parent_match = 0
    n_total = 0

    for ex in examples:
        symbols = ex["symbols"]
        tree_labels = ex["tree"]
        latex = ex["latex"]
        n = len(symbols)

        # Ground truth tree
        gt_roots, names = _build_trees(symbols, tree_labels)
        gt_latex = tree_to_latex(gt_roots)
        gt_svg = _dot_to_svg(_build_dot(gt_roots, names, "gt"))

        # Predicted tree
        if exhaustive:
            pred_roots, _ = predict_tree_exhaustive(
                model, symbol_vocab, symbols, device,
                max_subset=max_subset,
            )
        else:
            pred_roots, _ = predict_tree_iterative(
                model, symbol_vocab, symbols, device,
                max_subset=max_subset,
            )
        pred_latex = tree_to_latex(pred_roots)
        pred_svg = _dot_to_svg(_build_dot(pred_roots, names, "pr"))

        # Compare parent assignments
        gt_parents = [(t["parent"], t["edge_type"]) for t in tree_labels]
        pred_nodes_flat = []

        def _collect_flat(node):
            pred_nodes_flat.append(node)
            for et, children in sorted(node.children.items()):
                for c in children:
                    _collect_flat(c)

        for r in pred_roots:
            _collect_flat(r)
        pred_nodes_flat.sort(key=lambda n: n.index)

        pred_by_idx = {node.index: node for node in pred_nodes_flat}
        correct = 0
        for i in range(n):
            gt_p, gt_e = gt_parents[i]
            if i in pred_by_idx:
                pr_p = pred_by_idx[i].parent
                pr_e = pred_by_idx[i].edge_type
                if gt_p == pr_p and gt_e == pr_e:
                    correct += 1
        acc = correct / n
        n_parent_match += correct
        n_total += n

        latex_match = pred_latex == gt_latex
        match_class = "match" if latex_match else "nomatch"
        match_icon = "&#10003;" if latex_match else "&#10007;"

        cards.append(f"""
        <div class="card">
          <div class="latex-col">
            <div class="section-label">Expression</div>
            <div class="rendered">$${_escape(latex)}$$</div>
            <div class="code"><code>{_escape(latex)}</code></div>
            <div class="glyphs">{n} glyphs | {acc:.0%} correct</div>
            <div class="section-label" style="margin-top:12px">Predicted LaTeX</div>
            <div class="roundtrip {match_class}">
              <span>{match_icon}</span>
              <code>{_escape(pred_latex)}</code>
            </div>
          </div>
          <div class="tree-col">
            <div class="section-label">Ground Truth</div>
            {gt_svg}
          </div>
          <div class="tree-col">
            <div class="section-label">Predicted</div>
            {pred_svg}
          </div>
        </div>
        """)

    overall_acc = n_parent_match / max(n_total, 1)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Subset Model Predictions</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background: #fafafa; margin: 20px; }}
  h1 {{ text-align: center; color: #333; }}
  .summary {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }}
  .legend {{
    display: flex; gap: 18px; justify-content: center;
    margin: 10px 0 25px; font-size: 13px;
  }}
  .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .legend .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .card {{
    display: flex; gap: 20px; align-items: flex-start;
    background: white; border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    padding: 24px; margin-bottom: 20px;
  }}
  .latex-col {{ flex: 0 0 280px; }}
  .tree-col {{ flex: 1; overflow-x: auto; }}
  .tree-col svg {{ max-width: 100%; height: auto; }}
  .section-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    color: #999; margin-bottom: 8px;
  }}
  .rendered {{ font-size: 20px; margin-bottom: 8px; }}
  .code {{ font-size: 11px; color: #555; word-break: break-all; margin-bottom: 6px; }}
  .glyphs {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
  .roundtrip {{ font-size: 11px; }}
  .roundtrip.match {{ color: #4CAF50; }}
  .roundtrip.nomatch {{ color: #F44336; }}
  .roundtrip code {{ color: inherit; }}
</style>
</head>
<body>
<h1>{"Exhaustive" if exhaustive else "Iterative"}: Predictions vs Ground Truth</h1>
<div class="summary">Overall parent+edge accuracy: {overall_acc:.1%} ({n_parent_match}/{n_total})</div>
<div class="legend">
  <span><span class="dot" style="background:{EDGE_COLORS[-1]}"></span> root</span>
  <span><span class="dot" style="background:{EDGE_COLORS[NUM]}"></span> num</span>
  <span><span class="dot" style="background:{EDGE_COLORS[DEN]}"></span> den</span>
  <span><span class="dot" style="background:{EDGE_COLORS[SUP]}"></span> sup</span>
  <span><span class="dot" style="background:{EDGE_COLORS[SUB]}"></span> sub</span>
  <span><span class="dot" style="background:{EDGE_COLORS[SQRT_CONTENT]}"></span> sqrt</span>
</div>
{"".join(cards)}
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Written to {out_path} ({overall_acc:.1%} overall accuracy)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default=os.path.join(os.path.dirname(__file__), "..", "data", "tree_val.jsonl"))
    parser.add_argument("--run", default="v3")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--filter", default=None, help="Only show examples whose LaTeX contains this string")
    parser.add_argument("--exhaustive", action="store_true", help="Exhaustive mode (spatial subsets, no conflict resolution)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for example selection")
    args = parser.parse_args()

    out = os.path.join(os.path.dirname(__file__), "..", "tree_predictions.html")
    generate_html(args.val, args.run, out, n_examples=args.n, seed=args.seed,
                  filter_latex=args.filter, exhaustive=args.exhaustive)
