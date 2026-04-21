"""Analyze subset model errors on validation data.

Runs the full inference pipeline (subsets → evidence → tree) and compares
against ground truth. Categorizes failures by edge type, symbol, and size.

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 scripts/diagnostics/analyze_subset_errors.py --run mixed_v8
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from mathnote_ocr.engine.checkpoint import load_checkpoint
from mathnote_ocr.latex_utils.relations import compute_features_from_bbox_list
from mathnote_ocr.tree_parser.costs import COST_STRATEGIES
from mathnote_ocr.tree_parser.evidence import aggregate_evidence_soft
from mathnote_ocr.tree_parser.subset_model import load_subset_model
from mathnote_ocr.tree_parser.subset_selection import make_spatial_subsets
from mathnote_ocr.tree_parser.tree import EDGE_NAMES, ROOT

log = logging.getLogger(__name__)


def _edge_name(idx: int) -> str:
    if 0 <= idx < len(EDGE_NAMES):
        return EDGE_NAMES[idx]
    return f"edge_{idx}"


def _build_flat_predictions(evidence, N):
    """Build flat parent/edge predictions from evidence (no tree building).

    Returns list of dicts with parent, edge_type per symbol.
    """
    parent_votes = evidence["parent_votes"]  # (N, N+1, E)

    cost_fn = COST_STRATEGIES["propagate"]
    parent_scores = cost_fn(evidence, N)

    preds = []
    for i in range(N):
        # Best parent
        best_parent = parent_scores[i].argmax().item()
        if best_parent == N:
            best_parent = ROOT
            et = -1
        else:
            et = parent_votes[i, best_parent].argmax().item()
        preds.append({"parent": best_parent, "edge_type": et})
    return preds


def evaluate(args):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Load model
    ckpt = load_checkpoint("tree_subset", args.run, device=device)
    model = load_subset_model(ckpt, device=device)
    symbol_vocab = ckpt["symbol_vocab"]
    max_subset = ckpt["config"]["max_symbols"]
    unk_id = symbol_vocab.get("<unk>", 1)

    # Load val data
    val_path = Path(args.val)
    log.info("Run:    %s", args.run)
    log.info("Val:    %s", val_path)
    log.info("Device: %s", device)

    examples = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if len(ex["symbols"]) >= 2:
                examples.append(ex)
    if args.max_examples:
        examples = examples[: args.max_examples]
    log.info("Loaded %d examples", len(examples))

    # Counters
    parent_correct = 0
    parent_total = 0
    edge_correct = 0
    edge_total = 0
    parent_errors_by_edge = Counter()
    edge_confusion = Counter()
    errors_by_n_symbols = defaultdict(
        lambda: {"parent_ok": 0, "parent_err": 0, "edge_ok": 0, "edge_err": 0}
    )
    parent_err_symbols = Counter()
    edge_err_symbols = Counter()
    error_examples = []

    for ex_idx, ex in enumerate(examples):
        symbols = ex["symbols"]
        tree = ex["tree"]
        N = len(symbols)
        names = [s["name"] for s in symbols]
        bboxes = [s["bbox"] for s in symbols]

        # Run subset model → evidence
        subsets = make_spatial_subsets(bboxes, max_subset)
        partial_outputs = []

        for subset_indices in subsets:
            n_sub = len(subset_indices)
            S = max_subset

            sub_ids = torch.zeros(S, dtype=torch.long, device=device)
            for i, gi in enumerate(subset_indices):
                sub_ids[i] = symbol_vocab.get(names[gi], unk_id)

            bbox_list = [bboxes[gi] for gi in subset_indices]
            geo, size_feats = compute_features_from_bbox_list(bbox_list, S)
            geo = geo.to(device)
            size_feats = size_feats.to(device)

            pad_mask = torch.ones(S, dtype=torch.bool, device=device)
            pad_mask[:n_sub] = False

            with torch.no_grad():
                out = model.forward(
                    sub_ids.unsqueeze(0),
                    geo.unsqueeze(0),
                    pad_mask.unsqueeze(0),
                    size_feats.unsqueeze(0),
                )
            out_cpu = {k: v[0].cpu() for k, v in out.items()}
            partial_outputs.append((subset_indices, out_cpu, n_sub))

        evidence = aggregate_evidence_soft(N, partial_outputs)
        preds = _build_flat_predictions(evidence, N)

        # Compare
        n_bucket = min(N, 20)
        has_error = False
        for i in range(N):
            gt_parent = tree[i]["parent"]
            gt_edge = tree[i]["edge_type"]
            pred_parent = preds[i]["parent"]
            pred_edge = preds[i]["edge_type"]

            # Parent accuracy
            parent_total += 1
            if pred_parent == gt_parent:
                parent_correct += 1
                errors_by_n_symbols[n_bucket]["parent_ok"] += 1
            else:
                errors_by_n_symbols[n_bucket]["parent_err"] += 1
                parent_errors_by_edge[_edge_name(gt_edge)] += 1
                parent_err_symbols[names[i]] += 1
                has_error = True

            # Edge accuracy (only for non-ROOT children with correct parent)
            if gt_parent != ROOT and gt_edge >= 0 and pred_parent == gt_parent:
                edge_total += 1
                if pred_edge == gt_edge:
                    edge_correct += 1
                    errors_by_n_symbols[n_bucket]["edge_ok"] += 1
                else:
                    errors_by_n_symbols[n_bucket]["edge_err"] += 1
                    edge_confusion[(_edge_name(gt_edge), _edge_name(pred_edge))] += 1
                    edge_err_symbols[names[i]] += 1
                    has_error = True

        if has_error and len(error_examples) < 50:
            error_examples.append(
                {
                    "idx": ex_idx,
                    "latex": ex.get("latex", ""),
                    "n": N,
                    "gt_tree": tree,
                    "pred_tree": preds,
                    "names": names,
                }
            )

        if (ex_idx + 1) % 200 == 0:
            log.info(
                "  %d/%d  parent=%.1f%%  edge=%.1f%%",
                ex_idx + 1,
                len(examples),
                100 * parent_correct / max(parent_total, 1),
                100 * edge_correct / max(edge_total, 1),
            )

    # ── Report ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("RESULTS (%d examples)", len(examples))
    log.info("=" * 60)
    log.info(
        "Parent accuracy: %d/%d = %.1f%%",
        parent_correct,
        parent_total,
        100 * parent_correct / max(parent_total, 1),
    )
    log.info(
        "Edge accuracy:   %d/%d = %.1f%%",
        edge_correct,
        edge_total,
        100 * edge_correct / max(edge_total, 1),
    )

    log.info("")
    log.info("── Parent errors by true edge type ──")
    for edge, count in parent_errors_by_edge.most_common(15):
        log.info("  %-15s  %d", edge, count)

    log.info("")
    log.info("── Edge confusion (true → pred) ──")
    for (true_e, pred_e), count in edge_confusion.most_common(15):
        log.info("  %-10s → %-10s  %d", true_e, pred_e, count)

    log.info("")
    log.info("── Parent errors by symbol ──")
    for sym, count in parent_err_symbols.most_common(15):
        log.info("  %-15s  %d", sym, count)

    log.info("")
    log.info("── Edge errors by symbol ──")
    for sym, count in edge_err_symbols.most_common(15):
        log.info("  %-15s  %d", sym, count)

    log.info("")
    log.info("── Accuracy by expression size ──")
    for n_sym in sorted(errors_by_n_symbols.keys()):
        d = errors_by_n_symbols[n_sym]
        p_total = d["parent_ok"] + d["parent_err"]
        e_total = d["edge_ok"] + d["edge_err"]
        p_acc = d["parent_ok"] / p_total if p_total else 0
        e_acc = d["edge_ok"] / e_total if e_total else 0
        log.info(
            "  N=%2d  parent=%.1f%% (%5d)  edge=%.1f%% (%5d)",
            n_sym,
            100 * p_acc,
            p_total,
            100 * e_acc,
            e_total,
        )

    # Save error examples
    if error_examples:
        out_path = Path(f"data/tmp/subset_errors_{args.run}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(error_examples, f, indent=2)
        log.info("Saved %d error examples to %s", len(error_examples), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="mixed_v8")
    parser.add_argument("--val", default="data/runs/tree_subset/mixed_v7b/val.jsonl")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    log_dir = Path("data/tmp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"analyze_errors_{args.run}.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.basicConfig(level=logging.INFO, format=fmt._fmt, datefmt=fmt.datefmt)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

    log.info("Log file: %s", log_path)
    evaluate(args)
