#!/usr/bin/env python3
"""WebSocket server for testing the symbol classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import base64
import io
import json

import websockets

from mathnote_ocr import config
from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.engine.renderer import render_strokes
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


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, default=None, help="Pipeline config name")
ap.add_argument("--run", default=None, help="Classifier run name")
_args = ap.parse_args()

cfg = load_config(_resolve_config(_args.config))
classifier_run = _args.run or get(cfg, "classifier.run", "v4")

# Load model at startup
print(f"Loading classifier (run={classifier_run})...")
classifier = SymbolClassifier(run=classifier_run, weights_dir=REPO_WEIGHTS)
print(f"Loaded. Classes: {classifier.label_names}")
print(f"Device: {classifier.device}\n")


async def handler(websocket):
    addr = websocket.remote_address
    print(f"[connect] {addr}")

    async for message in websocket:
        try:
            msg = json.loads(message)

            if msg["type"] == "predict":
                raw_strokes = msg.get("strokes", [])
                if not raw_strokes:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No strokes to classify.",
                            }
                        )
                    )
                    continue

                stroke_width = msg.get("stroke_width", config.RENDER_STROKE_WIDTH)

                # Convert and render
                source_size = max(msg.get("canvas_width", 800), msg.get("canvas_height", 400))
                strokes = [Stroke.from_dicts(pts, id=i) for i, pts in enumerate(raw_strokes)]
                image = render_strokes(strokes, stroke_width=stroke_width, source_size=source_size)

                # Classify
                result, all_predictions = classifier.classify_topn(image, n=10)

                # Encode rendered image
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                rendered_b64 = base64.b64encode(buf.getvalue()).decode()

                # The raw predicted symbol (even if OOD-rejected)
                raw_symbol = all_predictions[0]["symbol"] if all_predictions else None

                print(
                    f"  predict: {result.symbol} (conf={result.confidence:.3f}, ood={result.is_ood})"
                )

                await websocket.send(
                    json.dumps(
                        {
                            "type": "prediction",
                            "symbol": result.symbol,
                            "raw_symbol": raw_symbol,
                            "confidence": result.confidence,
                            "prototype_distance": result.prototype_distance,
                            "is_ood": result.is_ood,
                            "rendered_image": f"data:image/png;base64,{rendered_b64}",
                            "all_predictions": all_predictions,
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
    print("Symbol Test Server")
    print("WebSocket: ws://localhost:8766\n")
    print("Open tools/test.html in your browser.\n")

    async with websockets.serve(handler, "localhost", 8766):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
