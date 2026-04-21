"""Evaluate backtrack tree builder on mixed_v7 val set."""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from mathnote_ocr.bbox import BBox
from mathnote_ocr.engine.grouper import DetectedSymbol
from mathnote_ocr.tree_parser.inference import SubsetTreeParser
from mathnote_ocr.tree_parser.tree_latex import tree_to_latex
from mathnote_ocr.tree_parser.tree_v2 import Tree, tree_from_arrays


def load_val(path, n_max=500):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n = len(ex["symbols"])
            if n < 2 or n > 20:
                continue
            examples.append(ex)
    if len(examples) > n_max:
        random.seed(42)
        examples = random.sample(examples, n_max)
    return examples


def gt_latex(ex):
    """Build ground truth tree and render to LaTeX."""
    return tree_to_latex(gt_to_tree(ex))


def gt_to_tree(ex):
    """Build ground truth Tree from example."""
    syms = ex["symbols"]
    tree_data = ex["tree"]
    names = [s["name"] for s in syms]
    bboxes = [s["bbox"] for s in syms]
    parent = [t["parent"] for t in tree_data]
    edge_type = [t["edge_type"] for t in tree_data]
    order = [t.get("order", 0) for t in tree_data]
    return tree_from_arrays(names, bboxes, parent, edge_type, order)


def trees_match(pred: Tree, gt: Tree) -> bool:
    """Compare two trees structurally: same parent, edge_type per symbol (by id)."""
    if len(pred.nodes) != len(gt.nodes):
        return False
    for sid in gt.nodes:
        if sid not in pred.nodes:
            return False
        pn = pred.nodes[sid]
        gn = gt.nodes[sid]
        if pn.symbol.name != gn.symbol.name:
            return False
        if pn.parent_id != gn.parent_id:
            return False
        if pn.edge_type != gn.edge_type:
            return False
    return True


def make_detected_symbols(ex):
    """Convert example to DetectedSymbol list."""
    symbols = []
    for s in ex["symbols"]:
        bbox = s["bbox"]
        ds = DetectedSymbol(
            stroke_indices=[],
            bbox=BBox(*bbox),
            symbol=s["name"],
            confidence=1.0,
            prototype_distance=0.0,
            alternatives=[],
        )
        symbols.append(ds)
    return symbols


REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIGS = REPO_ROOT / "configs"
REPO_WEIGHTS = REPO_ROOT / "weights"


def main():
    import argparse

    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument(
        "--config",
        default="mixed_v9_backtrack",
        help="Config name (looked up in repo configs/) or explicit path",
    )
    parser_arg.add_argument("--n", type=int, default=100)
    parser_arg.add_argument("--val", default=None, help="Path to val.jsonl (default: mixed_v7)")
    args = parser_arg.parse_args()

    # Resolve config: bundled name, repo config name, or path
    from mathnote_ocr.pipeline_config import load_config

    cfg_arg = args.config
    if "/" not in cfg_arg and not cfg_arg.endswith((".yaml", ".yml")):
        repo_config = REPO_CONFIGS / f"{cfg_arg}.yaml"
        if repo_config.exists():
            cfg_arg = str(repo_config)
    config = load_config(cfg_arg)
    tp_cfg = config.get("tree_parser", {})

    # Build parser (use repo weights so all experiment runs are available)
    tree_parser = SubsetTreeParser(
        subset_run=tp_cfg.get("subset_run", "mixed_v9"),
        max_subset=tp_cfg.get("max_subset", 8),
        max_iters=tp_cfg.get("max_iters", 3),
        scoring=tp_cfg.get("scoring", "full_spatial"),
        subset_strategy=tp_cfg.get("subset_strategy", "spatial"),
        tree_strategy=tp_cfg.get("tree_strategy", "backtrack"),
        tta_runs=tp_cfg.get("tta_runs", 1),
        tta_dx=tp_cfg.get("tta_dx", 0.05),
        tta_dy=tp_cfg.get("tta_dy", 0.15),
        tta_size=tp_cfg.get("tta_size", 0.05),
        root_discount=tp_cfg.get("root_discount", 0.2),
        weights_dir=str(REPO_WEIGHTS),
    )

    # Load GNN if configured
    gnn_run = tp_cfg.get("gnn_run")
    if gnn_run:
        from mathnote_ocr.engine.checkpoint import load_checkpoint
        from mathnote_ocr.tree_parser.gnn.model import EvidenceGNN

        gnn_ckpt = load_checkpoint(
            "tree_gnn", gnn_run, device=tree_parser.device, weights_dir=str(REPO_WEIGHTS)
        )
        gnn_cfg = gnn_ckpt["config"]
        gnn_model = EvidenceGNN(
            num_symbols=gnn_cfg["num_symbols"],
            d_model=gnn_cfg["d_model"],
            n_heads=gnn_cfg["n_heads"],
            n_layers=gnn_cfg["n_layers"],
            d_ff=gnn_cfg["d_ff"],
            d_arc=gnn_cfg["d_arc"],
            max_symbols=gnn_cfg.get("max_symbols", 64),
            d_edge=gnn_cfg.get("d_edge", 64),
        ).to(tree_parser.device)
        gnn_model.load_state_dict(gnn_ckpt["model_state_dict"])
        gnn_model.eval()
        tree_parser.gnn_model = gnn_model

    # Load val
    if args.val:
        val_path = args.val
    else:
        val_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "runs", "tree_subset", "mixed_v7", "val.jsonl"
        )
    examples = load_val(val_path, args.n)
    print(f"Evaluating {len(examples)} examples with config={args.config}")

    correct = 0
    errors = 0
    for i, ex in enumerate(examples):
        gt_tree = gt_to_tree(ex)
        detected = make_detected_symbols(ex)
        try:
            pred_latex, conf, pred_tree, _ = tree_parser.parse_with_tree(detected)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            errors += 1
            continue
        if trees_match(pred_tree, gt_tree):
            correct += 1
        elif i < 20 or (i % 50 == 0):
            print(f"  [{i}] WRONG: pred={tree_to_latex(pred_tree)}  gt={tree_to_latex(gt_tree)}")

    print(
        f"\nTree accuracy: {correct}/{len(examples)} = {correct / len(examples) * 100:.1f}% (errors: {errors})"
    )


if __name__ == "__main__":
    main()
