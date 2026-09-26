"""Check the skip fallback on real failures.

Re-creates the 19 empty detections from before the 2026-09-26 grouper
fixes (both fixes disabled in-process) and prints, for each, which
strokes the fallback leaves unexplained (with their GT label) and the
partial result. Expected: none empty; the unexplained stroke belongs to
the symbol that caused the failure.

Usage:
    python3.10 scripts/diagnostics/fallback_on_old_failures.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/model_evaluation")
from eval_handwritten_e2e import load_examples, strokes_from_example, gt_stroke_labels
from mathnote_ocr import MathOCR
from mathnote_ocr.engine import grouper

grouper._singletons_can_coexist = lambda *a, **k: True                       # undo fix 1
grouper._symbols_clash = lambda na, ba, nb, bb: "sqrt" not in (na, nb) and grouper._symbols_conflict(ba, bb)  # undo fix 2

ocr = MathOCR(config="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
old_empty = [("run_001", i) for i in (11, 14, 15, 27, 43, 51, 69, 79, 80)] + [("run_002", 75), ("run_002", 141)] \
    + [("run_003", i) for i in (2, 7, 9, 50, 55, 56, 59, 60)]
still_empty = 0
for run, i in old_empty:
    ex = load_examples(Path("data/shared/tree_handwritten") / run / "train_strokes.jsonl")[i]
    e = ocr.detect(strokes_from_example(ex), canvas_size=max(ex["canvas_width"], ex["canvas_height"]))
    gt = gt_stroke_labels(ex)
    still_empty += len(e) == 0
    print(f"{run}#{i:<3} unexplained={[(s, gt[s]) for s in e.unexplained_stroke_ids]}  partial: {e.latex[:55]}")
print(f"\n{len(old_empty)} old failures, {still_empty} still empty")
