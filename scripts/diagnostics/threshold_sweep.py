"""Cost of a stricter classifier.min_confidence on real handwriting.

With the exact-cover skip fallback, a rejected real symbol no longer
empties the whole expression — its strokes come back unexplained (red in
the app). This sweep measures, per threshold, how many REAL strokes end
up unexplained or mislabeled. Pair it with a garbage-drawing set to see
the benefit side.

Usage:
    python3.10 scripts/diagnostics/threshold_sweep.py
    python3.10 scripts/diagnostics/threshold_sweep.py --thresholds 0.13 0.3 0.5
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_evaluation"))
from eval_handwritten_e2e import gt_stroke_labels, load_examples, strokes_from_example  # noqa: E402

from mathnote_ocr import MathOCR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../math_ocr_web/configs/mixed_v10_backtrack_gnn.yaml")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.13, 0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--runs", nargs="+", default=["run_001", "run_002", "run_003"])
    ap.add_argument("--data-dir", default="data/shared/tree_handwritten")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    examples = [
        ex for run in args.runs
        for ex in load_examples(Path(args.data_dir) / run / "train_strokes.jsonl")
        if strokes_from_example(ex)
    ]

    print(f"{len(examples)} expressions\n")
    print(f"{'min_conf':>8} {'per-stroke':>11} {'unexplained':>12} {'exprs w/ red':>13} {'empty':>6}")
    for th in args.thresholds:
        cfg = {**base_cfg, "classifier": {**base_cfg.get("classifier", {}), "min_confidence": th}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
        ocr = MathOCR(config=f.name)

        n = correct = unexplained = exprs_red = empty = 0
        for ex in examples:
            expr = ocr.detect(strokes_from_example(ex),
                              canvas_size=max(ex.get("canvas_width", 800), ex.get("canvas_height", 800)))
            gt = gt_stroke_labels(ex)
            pred = {st.id: sym.name for sym in expr for st in sym.strokes}
            n += len(gt)
            correct += sum(pred.get(sid) == name for sid, name in gt.items())
            unexplained += len(expr.unexplained_stroke_ids)
            exprs_red += bool(expr.unexplained_stroke_ids)
            empty += len(expr) == 0
        print(f"{th:>8.2f} {100 * correct / n:>10.1f}% {100 * unexplained / n:>11.1f}% "
              f"{exprs_red:>13d} {empty:>6d}", flush=True)


if __name__ == "__main__":
    main()
