#!/usr/bin/env python3
"""WebSocket server for symbol data collection."""

from pathlib import Path

import asyncio
import base64
import io
import json
import re

import websockets

from mathnote_ocr import config
from mathnote_ocr.engine.renderer import render_strokes
from mathnote_ocr.engine.stroke import Stroke

# Set by CLI at startup
SYMBOLS_DIR: Path = Path("data/shared/symbols")


def get_next_id(label_dir: Path) -> str:
    """Find the next available 4-digit ID in label_dir."""
    existing = sorted(label_dir.glob("*.png"))
    if not existing:
        return "0001"

    max_num = 0
    for f in existing:
        match = re.match(r"(\d+)", f.stem)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"{max_num + 1:04d}"


def get_all_counts() -> dict[str, int]:
    """Count PNG files per class in the symbols directory."""
    counts = {}
    if SYMBOLS_DIR.exists():
        for d in sorted(SYMBOLS_DIR.iterdir()):
            if d.is_dir():
                counts[d.name] = len(list(d.glob("*.png")))
    return counts


async def handler(websocket):
    addr = websocket.remote_address
    print(f"[connect] {addr}")
    current_label = ""

    async for message in websocket:
        try:
            msg = json.loads(message)
            msg_type = msg.get("type")

            if msg_type == "set_label":
                current_label = msg["label"].strip()
                label_dir = SYMBOLS_DIR / current_label
                count = len(list(label_dir.glob("*.png"))) if label_dir.exists() else 0
                await websocket.send(
                    json.dumps(
                        {
                            "type": "label_set",
                            "label": current_label,
                            "count": count,
                        }
                    )
                )

            elif msg_type == "save":
                if not current_label:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No label set. Select a label first.",
                            }
                        )
                    )
                    continue

                raw_strokes = msg.get("strokes", [])
                if not raw_strokes:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No strokes to save.",
                            }
                        )
                    )
                    continue

                stroke_width = msg.get("stroke_width", config.RENDER_STROKE_WIDTH)

                # Convert to Stroke objects and render
                source_size = max(msg.get("canvas_width", 800), msg.get("canvas_height", 400))
                strokes = [Stroke.from_dicts(pts, id=i) for i, pts in enumerate(raw_strokes)]
                image = render_strokes(strokes, stroke_width=stroke_width, source_size=source_size)

                # Save PNG
                label_dir = SYMBOLS_DIR / current_label
                label_dir.mkdir(parents=True, exist_ok=True)
                file_id = get_next_id(label_dir)
                png_path = label_dir / f"{file_id}.png"
                image.save(png_path)

                # Save stroke JSON
                json_path = label_dir / f"{file_id}.json"
                stroke_data = {
                    "strokes": raw_strokes,
                    "canvas_width": msg.get("canvas_width", 800),
                    "canvas_height": msg.get("canvas_height", 400),
                    "stroke_width": stroke_width,
                    "label": current_label,
                }
                json_path.write_text(json.dumps(stroke_data))

                # Encode rendered image for client preview
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                rendered_b64 = base64.b64encode(buf.getvalue()).decode()

                count = len(list(label_dir.glob("*.png")))
                await websocket.send(
                    json.dumps(
                        {
                            "type": "saved",
                            "count": count,
                            "rendered_image": f"data:image/png;base64,{rendered_b64}",
                        }
                    )
                )
                print(f"  Saved {current_label}/{file_id} (total: {count})")

            elif msg_type == "get_counts":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "counts",
                            "counts": get_all_counts(),
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


async def run_server(port: int):
    print("Symbol Collection Server")
    print(f"Saving to: {SYMBOLS_DIR}")
    print(f"WebSocket: ws://localhost:{port}\n")
    print("Open tools/collect.html in your browser.\n")

    counts = get_all_counts()
    if counts:
        print("Current counts:")
        for label, count in counts.items():
            print(f"  {label}: {count}")
        print()

    async with websockets.serve(handler, "localhost", port):
        await asyncio.Future()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Symbol collection server")
    ap.add_argument(
        "--output-dir",
        default="data/shared/symbols",
        help="Directory to save collected symbols (default: data/shared/symbols)",
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    SYMBOLS_DIR = Path(args.output_dir).resolve()
    asyncio.run(run_server(args.port))
