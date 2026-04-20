"""FastAPI server for math OCR pipeline.

Usage:
    cd math_ocr_v2
    PYTHONPATH=. python3.10 tools/web/server.py
    PYTHONPATH=. python3.10 tools/web/server.py --config mixed_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from mathnote_ocr.engine.stroke import Stroke
from mathnote_ocr.engine.grouper import group_and_classify, GrouperCache, GrouperParams
from mathnote_ocr.classifier.inference import SymbolClassifier
from mathnote_ocr.tree_parser.inference import SubsetTreeParser
from mathnote_ocr.tree_parser.tree_v2 import Edge, ROOT_ID
from mathnote_ocr.pipeline_config import load_config, get
import mathnote_ocr.config as app_config

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"


def _resolve_config(name: str | None) -> str | None:
    """Resolve a bare config name to the repo's configs/ dir. Leave paths alone."""
    if name is None:
        return None
    if "/" in name or name.endswith((".yaml", ".yml")):
        return name
    repo_path = CONFIGS_DIR / f"{name}.yaml"
    if repo_path.exists():
        return str(repo_path)
    return name  # fall back to bundled


# ── Pipeline state ───────────────────────────────────────────────────

class Pipeline:
    """Holds loaded models and config. Can be reloaded."""

    def __init__(self, config_name: str | None = "default"):
        self.config_name = config_name
        self.cfg = load_config(_resolve_config(config_name))
        self._load()

    def _load(self):
        cfg = self.cfg

        classifier_run = get(cfg, "classifier.run", "v9_combined")
        tree_run = get(cfg, "tree_parser.subset_run", "mixed_v8")
        tree_gnn_run = get(cfg, "tree_parser.gnn_run")
        scoring = get(cfg, "tree_parser.scoring", "full_spatial")

        log.info("Loading classifier (run=%s)...", classifier_run)
        self.classifier = SymbolClassifier(
            run=classifier_run,
            ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
            per_class_thresholds=get(cfg, "classifier.per_class_thresholds", {}),
            weights_dir=str(WEIGHTS_DIR),
        )

        self.grouper_params = GrouperParams(
            max_strokes_per_symbol=get(cfg, "grouper.max_strokes_per_symbol", 4),
            size_multiplier=get(cfg, "grouper.size_multiplier", 0.1),
            min_merge_distance=get(cfg, "grouper.min_merge_distance", 14.0),
            max_group_diameter_ratio=get(cfg, "grouper.max_group_diameter_ratio", 2.2),
            conflict_threshold=get(cfg, "grouper.conflict_threshold", 0.32),
            min_confidence=get(cfg, "classifier.min_confidence", 0.15),
            ood_threshold=get(cfg, "classifier.ood_threshold", 15.0),
            stroke_width=get(cfg, "grouper.stroke_width", app_config.RENDER_STROKE_WIDTH),
        )
        self.top_k = get(cfg, "grouper.top_k", 10)

        subset_strategy = get(cfg, "tree_parser.subset_strategy", "spatial")
        tree_strategy = get(cfg, "tree_parser.tree_strategy", "edmonds")
        tree_kwargs = dict(
            subset_run=tree_run,
            scoring=scoring,
            subset_strategy=subset_strategy,
            k_neighbors=get(cfg, "tree_parser.k_neighbors", 5),
            subset_min_size=get(cfg, "tree_parser.subset_min_size", 2),
            subset_max_size=get(cfg, "tree_parser.subset_max_size", 5),
            xaxis_width_mults=get(cfg, "tree_parser.xaxis_width_mults"),
            tree_strategy=tree_strategy,
            seq_threshold=get(cfg, "tree_parser.seq_threshold", 0.7),
            tta_runs=get(cfg, "tree_parser.tta_runs", 1),
            tta_dx=get(cfg, "tree_parser.tta_dx", 0.05),
            tta_dy=get(cfg, "tree_parser.tta_dy", 0.15),
            tta_size=get(cfg, "tree_parser.tta_size", 0.05),
            spatial_penalty=get(cfg, "tree_parser.spatial_penalty", 3.0),
            root_discount=get(cfg, "tree_parser.root_discount", 0.2),
            weights_dir=str(WEIGHTS_DIR),
        )

        if tree_gnn_run:
            from mathnote_ocr.tree_parser.inference import GNNTreeParser
            log.info("Loading tree parser (run=%s, gnn=%s, strategy=%s)...",
                     tree_run, tree_gnn_run, subset_strategy)
            self.parser = GNNTreeParser(gnn_run=tree_gnn_run, **tree_kwargs)
        else:
            log.info("Loading tree parser (run=%s, strategy=%s)...",
                     tree_run, subset_strategy)
            self.parser = SubsetTreeParser(**tree_kwargs)

        log.info("Pipeline loaded.")

    def reload(self, config_name: str):
        """Reload with a different config."""
        self.config_name = config_name
        self.cfg = load_config(_resolve_config(config_name))
        self._load()

    def get_config_summary(self) -> dict:
        return {
            "config_name": self.config_name,
            "classifier_run": get(self.cfg, "classifier.run", "v9_combined"),
            "tree_run": get(self.cfg, "tree_parser.subset_run", "mixed_v8"),
            "gnn_run": get(self.cfg, "tree_parser.gnn_run"),
            "scoring": get(self.cfg, "tree_parser.scoring", "full_spatial"),
            "grouper_params": {
                "stroke_width": self.grouper_params.stroke_width,
                "min_confidence": self.grouper_params.min_confidence,
                "ood_threshold": self.grouper_params.ood_threshold,
                "top_k": self.top_k,
            },
        }


# ── App setup ────────────────────────────────────────────────────────

def create_app(config_name: str | None = "default") -> FastAPI:
    app = FastAPI(title="Math OCR")
    pipeline = Pipeline(config_name)

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    # ── REST endpoints ───────────────────────────────────────────────

    @app.get("/")
    async def index():
        html = (WEB_DIR / "templates" / "parse.html").read_text()
        return HTMLResponse(html)

    @app.get("/api/config")
    async def get_config():
        return JSONResponse(pipeline.get_config_summary())

    @app.get("/api/configs")
    async def list_configs():
        configs = [p.stem for p in CONFIGS_DIR.glob("*.yaml")]
        return JSONResponse({"configs": sorted(configs), "current": pipeline.config_name})

    @app.post("/api/config/{name}")
    async def set_config(name: str):
        try:
            pipeline.reload(name)
            return JSONResponse({"status": "ok", "config": pipeline.get_config_summary()})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    @app.get("/api/runs")
    async def list_runs():
        runs = {}
        for model_dir in WEIGHTS_DIR.iterdir():
            if model_dir.is_dir():
                model_runs = [d.name for d in model_dir.iterdir() if d.is_dir() and (d / "checkpoint.pth").exists()]
                if model_runs:
                    runs[model_dir.name] = sorted(model_runs)
        return JSONResponse(runs)

    # ── WebSocket (real-time drawing) ────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        cache = GrouperCache()
        log.info("WebSocket connected")

        try:
            while True:
                message = await websocket.receive_text()
                msg = json.loads(message)

                if msg["type"] == "clear":
                    cache.clear()
                    continue

                if msg["type"] == "detect":
                    raw_strokes = msg.get("strokes", [])
                    cache.update(len(raw_strokes))

                    if not raw_strokes:
                        await websocket.send_json({"type": "error", "message": "No strokes."})
                        continue

                    stroke_width = msg.get("stroke_width", pipeline.grouper_params.stroke_width)
                    source_size = max(
                        msg.get("canvas_width", 800),
                        msg.get("canvas_height", 400),
                    )

                    strokes = [Stroke.from_dicts(pts) for pts in raw_strokes]

                    t0 = time.perf_counter()
                    all_partitions = group_and_classify(
                        strokes, pipeline.classifier,
                        stroke_width=stroke_width,
                        source_size=source_size,
                        top_k=pipeline.top_k,
                        debug=msg.get("debug", False),
                        cache=cache,
                        params=pipeline.grouper_params,
                    )
                    t_group = time.perf_counter() - t0

                    if not all_partitions:
                        await websocket.send_json({"type": "result", "partitions": []})
                        continue

                    # Dump detected symbols for debugging
                    if msg.get("debug"):
                        best_partition = max(all_partitions, key=lambda p: sum(s.confidence for s in p))
                        symbols_dump = [
                            {"symbol": s.symbol, "bbox": [s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h],
                             "confidence": round(s.confidence, 3)}
                            for s in sorted(best_partition, key=lambda s: s.bbox.x)
                        ]
                        import json as _json
                        log.info("SYMBOLS: %s", _json.dumps(symbols_dump))

                    t1 = time.perf_counter()
                    debug_mode = msg.get("debug", False)
                    result_partitions = _build_results(
                        all_partitions, pipeline.parser,
                        diagnostics=debug_mode,
                    )
                    t_tree = time.perf_counter() - t1

                    log.info(
                        "%d strokes -> %d results  grouper=%dms  tree=%dms",
                        len(raw_strokes), len(result_partitions),
                        int(t_group * 1000), int(t_tree * 1000),
                    )
                    for j, r in enumerate(result_partitions[:3]):
                        log.info("  [%d] %s (score=%s)", j, r["latex"], r["score"])

                    await websocket.send_json({
                        "type": "result",
                        "partitions": result_partitions,
                        "timing": {
                            "grouper_ms": round(t_group * 1000),
                            "tree_ms": round(t_tree * 1000),
                        },
                    })

        except WebSocketDisconnect:
            log.info("WebSocket disconnected")
        except Exception as e:
            log.exception("WebSocket error")
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    return app


def _build_results(all_partitions, parser, diagnostics: bool = False) -> list[dict]:
    """Parse partitions into result dicts."""
    import torch

    result_partitions = []
    for partition in all_partitions:
        symbols = sorted(partition, key=lambda s: s.bbox.x)
        symbols = parser.promote_symbols(symbols)

        subset_diags = None
        if diagnostics and hasattr(parser, 'parse_with_diagnostics'):
            diag = parser.parse_with_diagnostics(symbols)
            latex = diag["latex"]
            parse_confidence = diag["confidence"]
            tree = diag["tree"]
            evidence = diag["evidence"]
            subset_diags = diag["subsets"]
        else:
            latex, parse_confidence, tree, evidence = parser.parse_with_tree(symbols)

        sym_confidence = 1.0
        for s in symbols:
            sym_confidence *= s.confidence
        sym_confidence = sym_confidence ** (1.0 / max(len(symbols), 1))
        combined = sym_confidence * parse_confidence

        # Build relations from tree
        relations = []
        if tree:
            parent_votes = evidence["parent_votes"] if evidence is not None else None
            N_syms = len(symbols)
            joint_probs = None
            if parent_votes is not None:
                N_pv, P1, E = parent_votes.shape
                flat = parent_votes.view(N_pv, -1)
                joint_probs = torch.softmax(flat, dim=-1).view(N_pv, P1, E)

            def _get_prob(child_idx, parent_idx, edge_type):
                if joint_probs is None:
                    return 1.0
                col = parent_idx if parent_idx >= 0 else N_syms
                if edge_type < 0:
                    return joint_probs[child_idx, col].sum().item()
                return joint_probs[child_idx, col, edge_type].item()

            def _walk(idx):
                for child_id, et, _ in tree.children_of(idx):
                    et_name = Edge(et).name.lower() if 0 <= et < len(Edge) else "root"
                    relations.append({
                        "from": child_id,
                        "to": idx,
                        "type": et_name,
                        "prob": round(_get_prob(child_id, idx, et), 3),
                    })
                    _walk(child_id)

            for root_idx in tree.root_ids():
                relations.append({
                    "from": root_idx,
                    "to": -1,
                    "type": "root",
                    "prob": round(_get_prob(root_idx, -1, -1), 3),
                })
                _walk(root_idx)

        entry = {
            "latex": latex,
            "parse_confidence": round(parse_confidence, 3),
            "symbol_confidence": round(sym_confidence, 3),
            "score": round(combined, 3),
            "symbols": [
                {
                    "symbol": s.symbol,
                    "confidence": s.confidence,
                    "stroke_indices": s.stroke_indices,
                    "bbox": {"x": s.bbox.x, "y": s.bbox.y, "w": s.bbox.w, "h": s.bbox.h},
                }
                for s in symbols
            ],
            "relations": relations,
        }
        if subset_diags is not None:
            entry["subsets"] = subset_diags
        result_partitions.append(entry)

    result_partitions.sort(key=lambda p: p["score"], reverse=True)
    return result_partitions


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description="Math OCR Web Server")
    ap.add_argument("--config", default="default", help="Pipeline config name")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--host", default="localhost")
    args = ap.parse_args()

    app = create_app(args.config)
    print(f"Running on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
