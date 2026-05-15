"""WebSocket server for interactive tree prediction browsing.

Loads the subset model and validation data at startup.
Clients request examples by index, with filtering.
Server runs predictions on demand and returns SVGs + LaTeX.

Usage:
    python3.10 tools/tree_browse_server.py --run v4 --votes
    # Then open tools/tree_browse.html in browser
"""

import asyncio
import json
import subprocess
from pathlib import Path

import torch
import websockets

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.evidence import aggregate_evidence, sample_subsets_with_coverage
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree import (
    DEN,
    EDGE_NAMES,
    NUM,
    SQRT_CONTENT,
    SUB,
    SUP,
    SymbolNode,
    build_tree,
    tree_to_latex,
)
from mathnote_ocr.tree_parser.tree_builder import build_tree_from_scores

REPO_WEIGHTS = str(Path(__file__).parent.parent / "weights")

EDGE_COLORS = {
    -1: "#666666",
    NUM: "#2196F3",
    DEN: "#F44336",
    SUP: "#4CAF50",
    SUB: "#FF9800",
    SQRT_CONTENT: "#9C27B0",
}


def _escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _dot_id(node, prefix="n"):
    return f"{prefix}{node.index}"


def _build_dot(roots, glyph_names, prefix="n"):
    lines = [
        "digraph T {",
        "  rankdir=TB;",
        '  node [shape=box, style="rounded,filled", fillcolor="#f5f5f5",'
        '        fontname="Courier", fontsize=14, margin="0.15,0.07"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    all_nodes = []

    def _collect(node):
        all_nodes.append(node)
        for et, children in sorted(node.children.items()):
            for c in children:
                _collect(c)

    for r in roots:
        _collect(r)

    for node in all_nodes:
        label = _escape(glyph_names[node.index])
        lines.append(f'  {_dot_id(node, prefix)} [label="{label}"];')

    sorted_roots = sorted(roots, key=lambda r: r.order)
    for i in range(1, len(sorted_roots)):
        prev_r = sorted_roots[i - 1]
        curr_r = sorted_roots[i]
        lines.append(
            f"  {_dot_id(prev_r, prefix)} -> {_dot_id(curr_r, prefix)}"
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
                    f"  {_dot_id(node, prefix)} -> {_dot_id(c, prefix)}"
                    f' [label=" {ename}", color="{color}",'
                    f'  fontcolor="{color}"];'
                )
            for i in range(1, len(sorted_children)):
                prev_c = sorted_children[i - 1]
                curr_c = sorted_children[i]
                lines.append(
                    f"  {_dot_id(prev_c, prefix)} -> {_dot_id(curr_c, prefix)}"
                    f' [style=dashed, color="#AAAAAA", constraint=false,'
                    f'  arrowsize=0.6, label=" seq", fontsize=8,'
                    f'  fontcolor="#AAAAAA"];'
                )

    lines.append("}")
    return "\n".join(lines)


def _dot_to_svg(dot_src):
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_src,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"<pre>graphviz error: {_escape(result.stderr)}</pre>"
    svg = result.stdout
    idx = svg.find("<svg")
    return svg[idx:] if idx >= 0 else svg


def _get_tags(latex):
    tags = []
    if "\\frac" in latex:
        tags.append("frac")
    if latex.count("\\frac") >= 2:
        tags.append("nested-frac")
    if "\\sqrt" in latex:
        tags.append("sqrt")
    if "^" in latex:
        tags.append("sup")
    if "_" in latex:
        tags.append("sub")
    return tags


# ── Global state ──────────────────────────────────────────────────────

model = None
symbol_vocab = None
max_subset = 8
device = torch.device("cpu")
use_votes = False
examples = []
filtered_indices = []  # indices into examples[]


def load_model(run_name):
    global model, symbol_vocab, max_subset
    ckpt = load_checkpoint("tree_subset", run_name, device=device, weights_dir=REPO_WEIGHTS)
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
    print(f"Model loaded: {cfg['num_symbols']} symbols, max_subset={max_subset}")


def load_data(val_path, max_symbols=20):
    global examples
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
            examples.append(ex)
    print(f"Loaded {len(examples)} validation examples")


def apply_filter(search="", tags=None):
    global filtered_indices
    tags = tags or []
    filtered_indices = []
    for i, ex in enumerate(examples):
        latex = ex["latex"]
        if search and search.lower() not in latex.lower():
            continue
        ex_tags = _get_tags(latex)
        if not all(t in ex_tags for t in tags):
            continue
        filtered_indices.append(i)


@torch.no_grad()
def predict_example(ex):
    symbols = ex["symbols"]
    tree_labels = ex["tree"]
    n = len(symbols)
    names = [s["name"] for s in symbols]

    # Ground truth
    gt_nodes = []
    for j, (s, t) in enumerate(zip(symbols, tree_labels)):
        gt_nodes.append(
            SymbolNode(
                symbol=s["name"],
                bbox=s["bbox"],
                index=j,
                parent=t["parent"],
                edge_type=t["edge_type"],
                order=t["order"],
            )
        )
    gt_roots = build_tree(gt_nodes)
    gt_latex = tree_to_latex(gt_roots)
    gt_svg = _dot_to_svg(_build_dot(gt_roots, names, "gt"))

    # Prediction
    if use_votes:
        bbox_list_all = [s["bbox"] for s in symbols]
        subsets = sample_subsets_with_coverage(
            n,
            bbox_list_all,
            n_subsets=50,
            min_size=3,
            max_size=min(max_subset, n),
        )
        partial_trees = []
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
            preds = model.predict(sub_ids, geo_buckets, sub_pad, size_feats, n_real=n_sub)
            partial_trees.append((subset_indices, preds))

        evidence = aggregate_evidence(n, partial_trees)
        parent_votes = evidence["parent_votes"]
        parent_scores = parent_votes.sum(dim=-1)
        root_counts = evidence["order_count"][:, n]
        parent_scores[:, n] = root_counts
        for i in range(n):
            parent_scores[i, i] = float("-inf")
        edge_type_scores = parent_votes
        order_sum = evidence["order_sum"]
        order_count = evidence["order_count"].clamp(min=1)
        order_preds = order_sum / order_count
    else:
        sub_ids = torch.zeros(n, dtype=torch.long, device=device)
        for i, s in enumerate(symbols):
            sub_ids[i] = symbol_vocab.get(s["name"], symbol_vocab.get("<unk>", 1))
        bbox_list = [s["bbox"] for s in symbols]
        geo_buckets, size_feats = compute_features_from_bbox_list(bbox_list, n)
        geo_buckets = geo_buckets.to(device)
        size_feats = size_feats.to(device)
        pad_mask = torch.zeros(n, dtype=torch.bool, device=device)
        out = model(
            sub_ids.unsqueeze(0),
            geo_buckets.unsqueeze(0),
            pad_mask.unsqueeze(0),
            size_feats.unsqueeze(0),
        )
        parent_scores = out["parent_scores"][0]
        edge_type_scores = out["edge_type_scores"][0]
        order_preds = out["order_preds"][0]

    bbox_lists = [s["bbox"] for s in symbols]
    pred_roots = build_tree_from_scores(
        parent_scores,
        edge_type_scores,
        order_preds,
        names,
        bbox_lists,
    )
    pred_latex = tree_to_latex(pred_roots)
    pred_svg = _dot_to_svg(_build_dot(pred_roots, names, "pr"))

    # Accuracy
    pred_flat = []

    def _collect(node):
        pred_flat.append(node)
        for et, children in sorted(node.children.items()):
            for c in children:
                _collect(c)

    for r in pred_roots:
        _collect(r)
    pred_by_idx = {node.index: node for node in pred_flat}

    correct = 0
    for i in range(n):
        gt_p, gt_e = tree_labels[i]["parent"], tree_labels[i]["edge_type"]
        if i in pred_by_idx:
            if pred_by_idx[i].parent == gt_p and pred_by_idx[i].edge_type == gt_e:
                correct += 1

    return {
        "latex": ex["latex"],
        "n": n,
        "gt_latex": gt_latex,
        "pred_latex": pred_latex,
        "gt_svg": gt_svg,
        "pred_svg": pred_svg,
        "acc": round(correct / n, 4),
        "latex_match": pred_latex == gt_latex,
        "tags": _get_tags(ex["latex"]),
    }


# ── WebSocket handler ─────────────────────────────────────────────────


async def handler(websocket):
    addr = websocket.remote_address
    print(f"[connect] {addr}")

    # Send initial info
    all_tags = sorted(set(t for ex in examples for t in _get_tags(ex["latex"])))
    apply_filter()
    await websocket.send(
        json.dumps(
            {
                "type": "init",
                "total": len(examples),
                "filtered": len(filtered_indices),
                "tags": all_tags,
            }
        )
    )

    async for message in websocket:
        try:
            msg = json.loads(message)

            if msg["type"] == "filter":
                search = msg.get("search", "")
                tags = msg.get("tags", [])
                apply_filter(search, tags)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "filter_result",
                            "filtered": len(filtered_indices),
                        }
                    )
                )

            elif msg["type"] == "get_page":
                page = msg.get("page", 0)
                per_page = msg.get("per_page", 5)
                start = page * per_page
                end = min(start + per_page, len(filtered_indices))

                items = []
                for idx in filtered_indices[start:end]:
                    items.append(predict_example(examples[idx]))

                await websocket.send(
                    json.dumps(
                        {
                            "type": "page",
                            "page": page,
                            "per_page": per_page,
                            "total": len(filtered_indices),
                            "items": items,
                        }
                    )
                )

        except Exception as e:
            import traceback

            traceback.print_exc()
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )
            )

    print(f"[disconnect] {addr}")


async def main():
    print("\nTree Browser Server")
    print("WebSocket: ws://localhost:8769")
    print("Open tools/tree_browse.html in your browser.\n")

    async with websockets.serve(handler, "localhost", 8769):
        await asyncio.Future()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val", default=str(Path(__file__).parent.parent / "data" / "tree_val.jsonl")
    )
    parser.add_argument("--run", default="v4")
    parser.add_argument("--votes", action="store_true")
    args = parser.parse_args()

    use_votes = args.votes

    print("Loading model...")
    load_model(args.run)
    print("Loading data...")
    load_data(args.val, max_symbols=20 if use_votes else max_subset)

    asyncio.run(main())
