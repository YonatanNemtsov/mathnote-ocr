"""Web demo for mathnote_ocr — draw math in the browser, get LaTeX back.

Uses only the public API: MathOCR and ocr_Session. Bundled defaults only.

    pip install -e .[tools]
    python demos/web/server.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from mathnote_ocr import MathOCR

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent


def create_app() -> FastAPI:
    app = FastAPI(title="MathNote OCR demo")
    ocr = MathOCR()
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/")
    async def index():
        return HTMLResponse((WEB_DIR / "templates" / "parse.html").read_text())

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        ocr_session = ocr.session()
        log.info("WebSocket connected")

        async def _send_result():
            t0 = time.perf_counter()
            expr = ocr_session.detect(top_k=5)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "%d strokes → %r (conf=%.2f) [%dms]",
                len(ocr_session), expr.latex, expr.confidence, elapsed_ms,
            )
            partitions = [expr.to_dict()] + [a.to_dict() for a in expr.alternatives]
            await websocket.send_json(
                {"type": "result", "partitions": partitions, "timing": {"total_ms": elapsed_ms}}
            )

        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                kind = msg.get("type")

                if kind == "add_stroke":
                    ocr_session.add_stroke(
                        msg["points"],
                        id=msg["id"],
                        width=msg.get("stroke_width", 2.0),
                    )
                    await _send_result()

                elif kind == "remove_stroke":
                    ocr_session.remove_stroke(msg["id"])
                    if len(ocr_session) > 0:
                        await _send_result()
                    else:
                        await websocket.send_json({"type": "result", "partitions": [], "timing": {"total_ms": 0}})

                elif kind == "clear":
                    ocr_session.clear()
                    await websocket.send_json({"type": "result", "partitions": [], "timing": {"total_ms": 0}})

                elif kind == "redetect":
                    ocr_session.clear()
                    sw = msg.get("stroke_width", 2.0)
                    stroke_ids = msg.get("stroke_ids", [])
                    for pts, sid in zip(msg.get("strokes", []), stroke_ids):
                        ocr_session.add_stroke(pts, id=sid, width=sw)
                    await _send_result()

        except WebSocketDisconnect:
            log.info("WebSocket disconnected")
        except Exception as e:
            log.exception("WebSocket error")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    app = create_app()
    print(f"Running on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
