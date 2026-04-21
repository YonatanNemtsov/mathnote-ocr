#!/usr/bin/env python3
"""WebSocket server for full math OCR pipeline: strokes → LaTeX."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import asyncio
import json
import time

import websockets

from mathnote_ocr import config  # for RENDER_STROKE_WIDTH default
from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.engine.grouper import GrouperCache, GrouperParams, group_and_classify
from mathnote_ocr.engine.stroke import Stroke
from mathnote_ocr.pipeline_config import get, load_config
from mathnote_ocr.tree_parser.inference import SubsetTreeParser
from mathnote_ocr.tree_parser.tree_v2 import Edge

REPO_ROOT = Path(__file__).parent.parent
REPO_CONFIGS = REPO_ROOT / "configs"
REPO_WEIGHTS = str(REPO_ROOT / "weights")


def _resolve_config(name):
    if name is None or "/" in name or name.endswith((".yaml", ".yml")):
        return name
    p = REPO_CONFIGS / f"{name}.yaml"
    return str(p) if p.exists() else name


# CLI — individual args override config values
ap = argparse.ArgumentParser(description="Math OCR Pipeline Server")
ap.add_argument(
    "--config", type=str, default=None, help="Pipeline config name (loads configs/{name}.yaml)"
)
ap.add_argument("--run", type=str, default=None, help="Tree subset model run name")
ap.add_argument(
    "--gnn-run", type=str, default=None, help="GNN model run name (enables GNN+iterative)"
)
ap.add_argument(
    "--gnn-simple", action="store_true", help="GNN without evidence anchoring or seq bonus"
)
ap.add_argument("--classifier-run", type=str, default=None, help="Classifier run name")
ap.add_argument(
    "--gnn-grouper",
    type=str,
    default=None,
    help="GNN grouper run (uses grouper_v2: GNN edges + CNN classify)",
)
ap.add_argument("--score-tree", type=str, default=None, help="Tree scoring method")
server_args = ap.parse_args()

cfg = load_config(_resolve_config(server_args.config))

# Resolve values: CLI > config > hardcoded default
classifier_run = server_args.classifier_run or get(cfg, "classifier.run", "v4")
grouper_type = get(cfg, "grouper.type", "classical")
grouper_gnn_run = server_args.gnn_grouper or get(cfg, "grouper.gnn_run")
grouper_classifier_run = get(cfg, "grouper.classifier_run", classifier_run)
top_k = get(cfg, "grouper.top_k", 10)
tree_run = server_args.run or get(cfg, "tree_parser.subset_run", "dg_all")
tree_gnn_run = server_args.gnn_run or get(cfg, "tree_parser.gnn_run")
score_tree = server_args.score_tree or get(cfg, "tree_parser.scoring", "full_spatial")

# Load models at startup
gnn_grouper = None
use_gnn_grouper = grouper_gnn_run or grouper_type == "gnn"
if use_gnn_grouper:
    from mathnote_ocr.engine.grouper_v2 import GNNGrouper

    gnn_run = grouper_gnn_run or get(cfg, "grouper.gnn_run", "v7")
    print(f"Loading GNN grouper (gnn={gnn_run}, classifier={grouper_classifier_run})...")
    gnn_grouper = GNNGrouper(
        gnn_run=gnn_run, classifier_run=grouper_classifier_run, weights_dir=REPO_WEIGHTS
    )
    classifier = gnn_grouper.classifier
    print(f"Loaded. Classes: {classifier.label_names}")
    print(f"Device: {classifier.device}")
else:
    print(f"Loading classifier (run={classifier_run})...")
    classifier = SymbolClassifier(
        run=classifier_run,
        ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
        per_class_thresholds=get(cfg, "classifier.per_class_thresholds", {}),
        weights_dir=REPO_WEIGHTS,
    )
    print(f"Loaded. Classes: {classifier.label_names}")
    print(f"Device: {classifier.device}")

# Build grouper params from config
grouper_params = GrouperParams(
    max_strokes_per_symbol=get(cfg, "grouper.max_strokes_per_symbol", 4),
    size_multiplier=get(cfg, "grouper.size_multiplier", 0.1),
    min_merge_distance=get(cfg, "grouper.min_merge_distance", 14.0),
    max_group_diameter_ratio=get(cfg, "grouper.max_group_diameter_ratio", 2.2),
    conflict_threshold=get(cfg, "grouper.conflict_threshold", 0.32),
    min_confidence=get(cfg, "classifier.min_confidence", 0.15),
    ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
    stroke_width=get(cfg, "grouper.stroke_width", config.RENDER_STROKE_WIDTH),
)

if tree_gnn_run:
    from mathnote_ocr.tree_parser.inference import GNNTreeParser

    anchor = not server_args.gnn_simple
    seq_bonus = not server_args.gnn_simple
    print(
        f"Loading tree parser (run={tree_run}, gnn={tree_gnn_run}, anchor={anchor}, seq={seq_bonus})..."
    )
    parser = GNNTreeParser(
        subset_run=tree_run,
        gnn_run=tree_gnn_run,
        anchor=anchor,
        seq_bonus=seq_bonus,
        scoring=score_tree,
        weights_dir=REPO_WEIGHTS,
    )
else:
    print(f"Loading tree parser (run={tree_run}, mode=subset)...")
    parser = SubsetTreeParser(subset_run=tree_run, scoring=score_tree, weights_dir=REPO_WEIGHTS)
print("Loaded.\n")


async def handler(websocket):
    addr = websocket.remote_address
    print(f"[connect] {addr}")
    cache = GrouperCache()

    async for message in websocket:
        try:
            msg = json.loads(message)

            if msg["type"] == "clear":
                cache.clear()
                continue

            if msg["type"] == "detect":
                raw_strokes = msg.get("strokes", [])
                cache.update(len(raw_strokes))
                if not raw_strokes:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No strokes to detect.",
                            }
                        )
                    )
                    continue

                stroke_width = msg.get("stroke_width", config.RENDER_STROKE_WIDTH)
                source_size = max(
                    msg.get("canvas_width", 800),
                    msg.get("canvas_height", 400),
                )

                strokes = [Stroke.from_dicts(pts) for pts in raw_strokes]

                t0 = time.perf_counter()
                if gnn_grouper:
                    all_partitions = gnn_grouper.group_and_classify(
                        strokes,
                        stroke_width=stroke_width,
                        source_size=source_size,
                        debug=msg.get("debug", False),
                    )
                else:
                    all_partitions = group_and_classify(
                        strokes,
                        classifier,
                        stroke_width=stroke_width,
                        source_size=source_size,
                        top_k=top_k,
                        debug=msg.get("debug", False),
                        cache=cache,
                        params=grouper_params,
                    )
                t_group = time.perf_counter() - t0

                if not all_partitions:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "result",
                                "partitions": [],
                            }
                        )
                    )
                    continue

                # Parse each partition
                t1 = time.perf_counter()
                result_partitions = []
                for partition in all_partitions:
                    symbols = sorted(partition, key=lambda s: s.bbox.x)
                    symbols = parser.promote_symbols(symbols)
                    latex, parse_confidence, tree, evidence = parser.parse_with_tree(symbols)

                    # Check for dropped symbols
                    detected = [s.symbol for s in symbols]
                    for sym_name in detected:
                        if sym_name not in latex and sym_name not in ("(", ")", "-", "+"):
                            mapped = {
                                "times": "\\times",
                                "dot": "\\cdot",
                                "sqrt": "\\sqrt",
                                "int": "\\int",
                                "sum": "\\sum",
                                "prod": "\\prod",
                            }.get(sym_name, sym_name)
                            if mapped not in latex:
                                print(
                                    f"  ⚠ DROPPED '{sym_name}' — detected: {detected} → latex: {latex}"
                                )

                    sym_confidence = 1.0
                    for s in symbols:
                        sym_confidence *= (
                            s.confidence
                        )  # already includes proto_quality from grouper
                    sym_confidence = sym_confidence ** (
                        1.0 / max(len(symbols), 1)
                    )  # geometric mean
                    combined = sym_confidence * parse_confidence

                    # Build relations from tree structure with probabilities
                    relations = []
                    if tree:
                        parent_votes = evidence["parent_votes"] if evidence is not None else None
                        # Joint (parent, edge_type) probabilities via softmax
                        # parent_votes shape: (N, N+1, E)
                        # Softmax over all (parent × edge_type) combos
                        import torch

                        joint_probs = None
                        N_syms = len(symbols)
                        if parent_votes is not None:
                            N_pv, P1, E = parent_votes.shape
                            flat = parent_votes.view(N_pv, -1)  # (N, (N+1)*E)
                            joint_probs = torch.softmax(flat, dim=-1).view(N_pv, P1, E)

                        def _get_prob(child_idx, parent_idx, edge_type):
                            if joint_probs is None:
                                return 1.0
                            col = parent_idx if parent_idx >= 0 else N_syms
                            if edge_type < 0:
                                # ROOT: sum across edge types for the ROOT column
                                return joint_probs[child_idx, col].sum().item()
                            return joint_probs[child_idx, col, edge_type].item()

                        def _walk(idx):
                            for child_id, et, _ in tree.children_of(idx):
                                et_name = Edge(et).name.lower() if 0 <= et < len(Edge) else "root"
                                relations.append(
                                    {
                                        "from": child_id,
                                        "to": idx,
                                        "type": et_name,
                                        "prob": round(_get_prob(child_id, idx, et), 3),
                                    }
                                )
                                _walk(child_id)

                        for root_idx in tree.root_ids():
                            relations.append(
                                {
                                    "from": root_idx,
                                    "to": -1,
                                    "type": "root",
                                    "prob": round(_get_prob(root_idx, -1, -1), 3),
                                }
                            )
                            _walk(root_idx)

                    result_partitions.append(
                        {
                            "latex": latex,
                            "parse_confidence": round(parse_confidence, 3),
                            "symbol_confidence": round(sym_confidence, 3),
                            "score": round(combined, 3),
                            "symbols": [
                                {
                                    "symbol": s.symbol,
                                    "confidence": s.confidence,
                                    "stroke_indices": s.stroke_indices,
                                    "bbox": {
                                        "x": s.bbox.x,
                                        "y": s.bbox.y,
                                        "w": s.bbox.w,
                                        "h": s.bbox.h,
                                    },
                                }
                                for s in symbols
                            ],
                            "relations": relations,
                        }
                    )

                # Sort by combined score
                result_partitions.sort(key=lambda p: p["score"], reverse=True)

                t_tree = time.perf_counter() - t1
                print(
                    f"  {len(raw_strokes)} strokes → {len(result_partitions)} results  "
                    f"grouper={t_group * 1000:.0f}ms  tree={t_tree * 1000:.0f}ms ({len(all_partitions)}p)"
                )
                for j, r in enumerate(result_partitions[:3]):
                    print(f"    [{j}] {r['latex']} (score={r['score']})")

                await websocket.send(
                    json.dumps(
                        {
                            "type": "result",
                            "partitions": result_partitions,
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
    print("Math OCR Pipeline Server")
    print("WebSocket: ws://localhost:8768\n")
    print("Open tools/parse.html in your browser.\n")

    async with websockets.serve(handler, "localhost", 8768):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
