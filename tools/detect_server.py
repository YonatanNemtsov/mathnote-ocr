#!/usr/bin/env python3
"""WebSocket server for expression detection (grouping + classification)."""

import asyncio
import json
from pathlib import Path

import websockets

from mathnote_ocr import config
from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.engine.grouper import group_and_classify
from mathnote_ocr.engine.layout import analyze_layout
from mathnote_ocr.engine.stroke import Stroke
from mathnote_ocr.pipeline_config import get, load_config

REPO_ROOT = Path(__file__).parent.parent
REPO_CONFIGS = REPO_ROOT / "configs"
REPO_WEIGHTS = str(REPO_ROOT / "weights")


def _resolve_config(name):
    if name is None or "/" in name or name.endswith((".yaml", ".yml")):
        return name
    p = REPO_CONFIGS / f"{name}.yaml"
    return str(p) if p.exists() else name


# Load model at startup
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default=None, help="Pipeline config name")
ap.add_argument("--run", default=None, help="Classifier run name")
_args = ap.parse_args()

cfg = load_config(_resolve_config(_args.config))
classifier_run = _args.run or get(cfg, "classifier.run", "v4")

print(f"Loading classifier (run={classifier_run})...")
classifier = SymbolClassifier(run=classifier_run, weights_dir=REPO_WEIGHTS)
print(f"Loaded. Classes: {len(classifier.label_names)}")
print(f"Device: {classifier.device}\n")


async def handler(websocket):
    addr = websocket.remote_address
    print(f"[connect] {addr}")
    async for message in websocket:
        try:
            msg = json.loads(message)

            if msg["type"] == "clear":
                continue

            if msg["type"] == "detect":
                raw_strokes = msg.get("strokes", [])
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

                strokes = [Stroke.from_dicts(pts, id=i) for i, pts in enumerate(raw_strokes)]

                all_partitions = group_and_classify(
                    strokes,
                    classifier,
                    stroke_width=stroke_width,
                    source_size=source_size,
                    top_k=10,
                )

                def serialize_partition(symbols):
                    sorted_syms = sorted(symbols, key=lambda s: s.bbox.x)
                    layout = analyze_layout(sorted_syms)
                    return {
                        "symbols": [
                            {
                                "symbol": s.symbol,
                                "confidence": s.confidence,
                                "prototype_distance": s.prototype_distance,
                                "stroke_indices": s.stroke_indices,
                                "bbox": {
                                    "x": s.bbox.x,
                                    "y": s.bbox.y,
                                    "w": s.bbox.w,
                                    "h": s.bbox.h,
                                },
                                "alternatives": [
                                    {"symbol": sym, "confidence": c}
                                    for sym, c in (s.alternatives or [])
                                ],
                            }
                            for s in sorted_syms
                        ],
                        "edges": [
                            {
                                "source": e.source,
                                "target": e.target,
                                "v_offset": round(e.v_offset, 3),
                                "h_offset": round(e.h_offset, 3),
                                "size_ratio": round(e.size_ratio, 3),
                                "overlap_v": round(e.overlap_v, 3),
                            }
                            for e in layout.edges
                        ],
                        "expression": " ".join(s.symbol for s in sorted_syms),
                    }

                results = [serialize_partition(p) for p in all_partitions]
                print(f"  detect: {len(raw_strokes)} strokes → {len(results)} partitions")
                for j, r in enumerate(results[:3]):
                    print(f"    [{j}] {r['expression']}")

                await websocket.send(
                    json.dumps(
                        {
                            "type": "detection",
                            "partitions": results,
                        }
                    )
                )

        except Exception as e:
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
    print("Expression Detection Server")
    print("WebSocket: ws://localhost:8767\n")
    print("Open tools/detect.html in your browser.\n")

    async with websockets.serve(handler, "localhost", 8767):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
