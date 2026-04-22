#!/usr/bin/env python3
"""Interactive tree parser tester.

Serves a local web page where you type LaTeX, see it rendered,
and see what the model predicts.

Usage:
    python3.10 tools/test_tree.py --run v5
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import torch
import ziamath as zm

from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.gnn.model import EvidenceGNN
from mathnote_ocr.tree_parser.subset_model import SubsetTreeModel
from mathnote_ocr.tree_parser.tree import EDGE_NAMES, tree_to_latex
from scripts.diagnostics.visualize_predictions import (
    predict_tree_exhaustive,
    predict_tree_gnn,
    predict_tree_iterative,
)

# ── Globals (set in main) ────────────────────────────────────────────

MODEL = None
GNN_MODEL = None
SYMBOL_VOCAB = None
DEVICE = "cpu"


def render_svg_b64(latex: str) -> str | None:
    """Render LaTeX to base64-encoded SVG."""
    try:
        m = zm.Math.fromlatex(r"\displaystyle " + latex)
        svg = m.svg()
        return base64.b64encode(svg.encode()).decode()
    except Exception:
        return None


def _tree_info(roots, names):
    """Build per-symbol info from tree roots."""
    sym_info = []
    for node in _flatten_nodes(roots):
        et = EDGE_NAMES[node.edge_type] if 0 <= node.edge_type < len(EDGE_NAMES) else "root"
        par = names[node.parent] if node.parent >= 0 else "ROOT"
        sym_info.append(
            {
                "idx": node.index,
                "name": node.symbol,
                "parent": par,
                "parent_idx": node.parent,
                "edge": et,
                "order": node.order,
            }
        )
    sym_info.sort(key=lambda x: x["idx"])
    return sym_info


def run_model(latex: str) -> dict:
    """Extract glyphs, run both models, return results."""
    glyphs = _extract_glyphs(latex)
    if glyphs is None:
        return {"error": "Failed to extract glyphs"}

    symbols = [{"name": g["name"], "bbox": g["bbox"]} for g in glyphs]
    names = [s["name"] for s in symbols]

    # Exhaustive (baseline)
    try:
        roots_ex, _ = predict_tree_exhaustive(
            MODEL,
            SYMBOL_VOCAB,
            symbols,
            DEVICE,
        )
        pred_ex = tree_to_latex(roots_ex)
    except Exception as e:
        return {"error": f"Exhaustive error: {e}", "glyphs": names}

    # Iterative (SEQ-conflict targeted)
    pred_iter = None
    roots_iter = None
    try:
        roots_iter, _ = predict_tree_iterative(
            MODEL,
            SYMBOL_VOCAB,
            symbols,
            DEVICE,
        )
        pred_iter = tree_to_latex(roots_iter)
    except Exception as e:
        pred_iter = f"Error: {e}"
        roots_iter = None

    # GNN
    pred_gnn = None
    roots_gnn = None
    if GNN_MODEL is not None:
        try:
            roots_gnn, _ = predict_tree_gnn(
                GNN_MODEL,
                MODEL,
                SYMBOL_VOCAB,
                symbols,
                DEVICE,
            )
            pred_gnn = tree_to_latex(roots_gnn)
        except Exception as e:
            pred_gnn = f"Error: {e}"
            roots_gnn = None

    return {
        "pred_latex": pred_ex,
        "pred_latex_iter": pred_iter,
        "pred_latex_gnn": pred_gnn,
        "glyphs": names,
        "tree": _tree_info(roots_ex, names),
        "tree_iter": _tree_info(roots_iter, names) if roots_iter else None,
        "tree_gnn": _tree_info(roots_gnn, names) if roots_gnn else None,
        "match": pred_ex == latex,
        "match_iter": pred_iter == latex if pred_iter else False,
        "match_gnn": pred_gnn == latex if pred_gnn else False,
    }


def _flatten_nodes(roots):
    """Collect all nodes from tree."""
    result = []

    def walk(node):
        result.append(node)
        for children in node.children.values():
            for c in children:
                walk(c)

    for r in roots:
        walk(r)
    return result


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tree Parser Tester</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
  h1 { font-size: 22px; color: #333; }
  .input-row { display: flex; gap: 8px; margin-bottom: 20px; }
  input[type=text] { flex: 1; font-family: monospace; font-size: 16px; padding: 8px 12px; border: 2px solid #ccc; border-radius: 6px; }
  input[type=text]:focus { outline: none; border-color: #4a90d9; }
  button { padding: 8px 20px; font-size: 15px; border: none; background: #4a90d9; color: white; border-radius: 6px; cursor: pointer; }
  button:hover { background: #3a7bc8; }
  .result { background: white; border-radius: 8px; padding: 20px; margin-top: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .render-row { display: flex; gap: 30px; align-items: center; margin: 16px 0; }
  .render-box { text-align: center; }
  .render-box img { max-height: 80px; }
  .label { font-size: 12px; color: #888; text-transform: uppercase; margin-bottom: 4px; }
  .match { color: #2ecc40; font-weight: bold; font-size: 18px; }
  .miss { color: #e74c3c; font-weight: bold; font-size: 18px; }
  .pred-latex { font-family: monospace; font-size: 14px; color: #555; margin: 8px 0; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
  th, td { padding: 4px 10px; text-align: left; border-bottom: 1px solid #eee; }
  th { color: #888; font-weight: 500; }
  .glyphs { font-family: monospace; font-size: 13px; color: #666; margin: 8px 0; }
  .error { color: #e74c3c; }
  .history { margin-top: 30px; }
  .history-item { background: white; border-radius: 6px; padding: 12px 16px; margin: 8px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.06); display: flex; align-items: center; gap: 12px; }
  .history-item .status { font-size: 18px; }
  .history-item code { font-size: 13px; color: #555; }
  .tree-viz { overflow-x: auto; margin: 4px 0; padding: 8px; background: #f8f8f8; border-radius: 6px; }
</style>
</head>
<body>
<h1>Tree Parser Tester</h1>
<form id="form">
  <div class="input-row">
    <input type="text" id="latex" placeholder="Enter LaTeX... e.g. \\frac{x^{2}}{y+1}" autofocus>
    <button type="submit">Test</button>
  </div>
</form>
<div id="result"></div>
<div id="history" class="history"></div>

<script>
const form = document.getElementById('form');
const input = document.getElementById('latex');
const resultDiv = document.getElementById('result');
const historyDiv = document.getElementById('history');

function drawTreeSVG(treeData) {
  if (!treeData || treeData.length === 0) return '';
  var EC = {num:'#4a90d9',den:'#e74c3c',sup:'#2ecc40',sub:'#f39c12',sqrt_content:'#9b59b6',upper:'#1abc9c',lower:'#e67e22',match:'#95a5a6'};
  var byIdx = {}, ch = {}, roots = [];
  for (var i = 0; i < treeData.length; i++) {
    var n = treeData[i]; byIdx[n.idx] = n;
    if (n.parent_idx < 0) { roots.push(n); }
    else { if (!ch[n.parent_idx]) ch[n.parent_idx] = []; ch[n.parent_idx].push(n); }
  }
  var eo = {num:0,den:1,sqrt_content:2,upper:3,lower:4,sup:5,sub:6,match:7};
  for (var k in ch) ch[k].sort(function(a,b){ return (eo[a.edge]||9)-(eo[b.edge]||9) || a.order-b.order; });
  roots.sort(function(a,b){ return a.order - b.order; });
  var nodeH = 22, gapX = 12, levelH = 55, padX = 10, padY = 20;
  function tw(name) { return Math.max(name.length * 7.2 + 14, 36); }
  function sw(idx) {
    var c = ch[idx] || [];
    if (c.length === 0) return tw(byIdx[idx].name);
    var w = 0;
    for (var i = 0; i < c.length; i++) { if (i > 0) w += gapX; w += sw(c[i].idx); }
    return Math.max(w, tw(byIdx[idx].name));
  }
  var pos = {};
  function layout(idx, left, depth) {
    var c = ch[idx] || [];
    var myW = tw(byIdx[idx].name);
    if (c.length === 0) { pos[idx] = {x: left + myW/2, y: depth * levelH}; return myW; }
    var totalW = 0;
    for (var i = 0; i < c.length; i++) { if (i > 0) totalW += gapX; totalW += layout(c[i].idx, left + totalW, depth + 1); }
    var fc = pos[c[0].idx], lc = pos[c[c.length-1].idx];
    pos[idx] = {x: (fc.x + lc.x) / 2, y: depth * levelH};
    return Math.max(totalW, myW);
  }
  var totalW = 0;
  for (var i = 0; i < roots.length; i++) { if (i > 0) totalW += gapX * 3; totalW += layout(roots[i].idx, totalW, 0); }
  var maxD = 0;
  for (var k in pos) maxD = Math.max(maxD, pos[k].y);
  var svgW = totalW + padX * 2, svgH = maxD + nodeH + padY * 2;
  var svg = '<svg width="' + svgW + '" height="' + svgH + '" xmlns="http://www.w3.org/2000/svg" style="font-family:monospace;font-size:11px">';
  for (var i = 0; i < treeData.length; i++) {
    var n = treeData[i];
    if (n.parent_idx < 0 || !pos[n.parent_idx] || !pos[n.idx]) continue;
    var p = pos[n.parent_idx], c = pos[n.idx], col = EC[n.edge] || '#999';
    var px = p.x + padX, py = p.y + padY + nodeH, cx = c.x + padX, cy = c.y + padY;
    svg += '<line x1="' + px + '" y1="' + py + '" x2="' + cx + '" y2="' + cy + '" stroke="' + col + '" stroke-width="1.5"/>';
    var mx = (px + cx) / 2, my = (py + cy) / 2;
    svg += '<text x="' + (mx+4) + '" y="' + (my-3) + '" fill="' + col + '" font-size="9" font-weight="bold">' + n.edge + '</text>';
  }
  for (var i = 0; i < treeData.length; i++) {
    var n = treeData[i], p = pos[n.idx];
    if (!p) continue;
    var w = tw(n.name), x = p.x + padX - w/2, y = p.y + padY;
    var name = n.name.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    svg += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + nodeH + '" rx="4" fill="white" stroke="#bbb" stroke-width="1"/>';
    svg += '<text x="' + (p.x + padX) + '" y="' + (y + nodeH/2 + 4) + '" text-anchor="middle" fill="#333">' + name + '</text>';
  }
  svg += '</svg>';
  return svg;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const latex = input.value.trim();
  if (!latex) return;

  resultDiv.innerHTML = '<p style="color:#888">Running...</p>';

  const resp = await fetch('/test', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'latex=' + encodeURIComponent(latex),
  });
  const data = await resp.json();

  let html = '<div class="result">';

  if (data.error) {
    html += '<p class="error">' + data.error + '</p>';
    if (data.glyphs) html += '<p class="glyphs">Glyphs: ' + data.glyphs.join(', ') + '</p>';
    html += '</div>';
    resultDiv.innerHTML = html;
    return;
  }

  // Renders
  html += '<div class="render-row">';
  if (data.input_svg) {
    html += '<div class="render-box"><div class="label">Input</div>';
    html += '<img src="data:image/svg+xml;base64,' + data.input_svg + '"></div>';
  }
  if (data.pred_svg) {
    const ex_cls = data.match ? 'match' : 'miss';
    html += '<div class="render-box"><div class="label">Exhaustive <span class="' + ex_cls + '">' + (data.match ? 'MATCH' : 'MISS') + '</span></div>';
    html += '<img src="data:image/svg+xml;base64,' + data.pred_svg + '"></div>';
  }
  if (data.pred_svg_iter) {
    const con_cls = data.match_iter ? 'match' : 'miss';
    html += '<div class="render-box"><div class="label">Iterative <span class="' + con_cls + '">' + (data.match_iter ? 'MATCH' : 'MISS') + '</span></div>';
    html += '<img src="data:image/svg+xml;base64,' + data.pred_svg_iter + '"></div>';
  }
  if (data.pred_svg_gnn) {
    const gnn_cls = data.match_gnn ? 'match' : 'miss';
    html += '<div class="render-box"><div class="label">GNN <span class="' + gnn_cls + '">' + (data.match_gnn ? 'MATCH' : 'MISS') + '</span></div>';
    html += '<img src="data:image/svg+xml;base64,' + data.pred_svg_gnn + '"></div>';
  }
  html += '</div>';

  // Predicted LaTeX
  html += '<div class="pred-latex">Exhaustive: <code>' + data.pred_latex + '</code></div>';
  if (data.pred_latex_iter) {
    html += '<div class="pred-latex">Iterative: <code>' + data.pred_latex_iter + '</code></div>';
  }
  if (data.pred_latex_gnn) {
    html += '<div class="pred-latex">GNN: <code>' + data.pred_latex_gnn + '</code></div>';
  }

  // Glyphs
  html += '<div class="glyphs">Glyphs (' + data.glyphs.length + '): ' + data.glyphs.join(', ') + '</div>';

  // Tree visualizations
  if (data.tree) {
    html += '<div style="margin:12px 0"><div class="label">Exhaustive Tree</div>';
    html += '<div class="tree-viz">' + drawTreeSVG(data.tree) + '</div></div>';
  }
  if (data.tree_iter) {
    html += '<div style="margin:12px 0"><div class="label">Iterative Tree</div>';
    html += '<div class="tree-viz">' + drawTreeSVG(data.tree_iter) + '</div></div>';
  }
  if (data.tree_gnn) {
    html += '<div style="margin:12px 0"><div class="label">GNN Tree</div>';
    html += '<div class="tree-viz">' + drawTreeSVG(data.tree_gnn) + '</div></div>';
  }

  // Tree tables side by side
  html += '<div style="display:flex;gap:20px">';
  if (data.tree) {
    html += '<div style="flex:1"><div class="label">Exhaustive tree</div>';
    html += '<table><tr><th>#</th><th>Sym</th><th>Parent</th><th>Edge</th><th>Ord</th></tr>';
    for (const s of data.tree) {
      html += '<tr><td>' + s.idx + '</td><td>' + s.name + '</td><td>' + s.parent + '</td><td>' + s.edge + '</td><td>' + s.order + '</td></tr>';
    }
    html += '</table></div>';
  }
  if (data.tree_iter) {
    html += '<div style="flex:1"><div class="label">Iterative tree</div>';
    html += '<table><tr><th>#</th><th>Sym</th><th>Parent</th><th>Edge</th><th>Ord</th></tr>';
    for (const s of data.tree_iter) {
      const diff = data.tree && data.tree[s.idx] && (s.parent !== data.tree[s.idx].parent || s.edge !== data.tree[s.idx].edge) ? ' style="background:#fff3cd"' : '';
      html += '<tr' + diff + '><td>' + s.idx + '</td><td>' + s.name + '</td><td>' + s.parent + '</td><td>' + s.edge + '</td><td>' + s.order + '</td></tr>';
    }
    html += '</table></div>';
  }
  if (data.tree_gnn) {
    html += '<div style="flex:1"><div class="label">GNN tree</div>';
    html += '<table><tr><th>#</th><th>Sym</th><th>Parent</th><th>Edge</th><th>Ord</th></tr>';
    for (const s of data.tree_gnn) {
      const diff = data.tree && data.tree[s.idx] && (s.parent !== data.tree[s.idx].parent || s.edge !== data.tree[s.idx].edge) ? ' style="background:#fff3cd"' : '';
      html += '<tr' + diff + '><td>' + s.idx + '</td><td>' + s.name + '</td><td>' + s.parent + '</td><td>' + s.edge + '</td><td>' + s.order + '</td></tr>';
    }
    html += '</table></div>';
  }
  html += '</div>';

  html += '</div>';
  resultDiv.innerHTML = html;

  // Add to history
  const ex_icon = data.match ? '&#x2714;' : '&#x2718;';
  const con_icon = data.match_iter ? '&#x2714;' : '&#x2718;';
  const gnn_icon = data.match_gnn ? '&#x2714;' : '&#x2718;';
  const histItem = document.createElement('div');
  histItem.className = 'history-item';
  histItem.innerHTML = '<span class="status ' + (data.match ? 'match' : 'miss') + '">' + ex_icon + '</span>'
    + '<span class="status ' + (data.match_iter ? 'match' : 'miss') + '">' + con_icon + '</span>'
    + '<span class="status ' + (data.match_gnn ? 'match' : 'miss') + '">' + gnn_icon + '</span>'
    + '<code>' + latex + '</code>';
  histItem.style.cursor = 'pointer';
  histItem.addEventListener('click', () => { input.value = latex; form.dispatchEvent(new Event('submit')); });
  historyDiv.insertBefore(histItem, historyDiv.firstChild);
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        params = parse_qs(body)
        latex = params.get("latex", [""])[0]

        result = run_model(latex)

        # Add rendered SVGs
        if "error" not in result:
            result["input_svg"] = render_svg_b64(latex)
            result["pred_svg"] = render_svg_b64(result["pred_latex"])
            pred_lin = result.get("pred_latex_iter")
            if pred_lin and not pred_lin.startswith("Error"):
                result["pred_svg_iter"] = render_svg_b64(pred_lin)
            pred_gnn = result.get("pred_latex_gnn")
            if pred_gnn and not pred_gnn.startswith("Error"):
                result["pred_svg_gnn"] = render_svg_b64(pred_gnn)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    global MODEL, GNN_MODEL, SYMBOL_VOCAB, DEVICE

    parser = argparse.ArgumentParser(description="Interactive tree parser tester")
    parser.add_argument("--run", default="dg_all", help="Subset model run name")
    parser.add_argument("--gnn-run", default=None, help="GNN model run name (e.g. v2)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    DEVICE = args.device

    # Load subset model
    weights_dir = Path(__file__).parent.parent / "weights" / "tree_subset" / args.run
    ckpt_path = weights_dir / "checkpoint.pth"
    print(f"Loading subset model from {ckpt_path}...")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    SYMBOL_VOCAB = ckpt["symbol_vocab"]
    cfg = ckpt["config"]
    print(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")
    print(f"  {len(SYMBOL_VOCAB)} symbols, d_model={cfg['d_model']}")

    MODEL = SubsetTreeModel(**cfg)
    MODEL.load_state_dict(ckpt["model_state_dict"], strict=False)
    MODEL.to(DEVICE).eval()

    # Load GNN model (optional)
    if args.gnn_run:
        gnn_dir = Path(__file__).parent.parent / "weights" / "tree_gnn" / args.gnn_run
        gnn_path = gnn_dir / "checkpoint.pth"
        print(f"Loading GNN model from {gnn_path}...")
        gnn_ckpt = torch.load(gnn_path, map_location=DEVICE, weights_only=False)
        gnn_cfg = gnn_ckpt["config"]
        GNN_MODEL = EvidenceGNN(**gnn_cfg).to(DEVICE)
        GNN_MODEL.load_state_dict(gnn_ckpt["model_state_dict"])
        GNN_MODEL.eval()
        print(f"  GNN loaded: d_model={gnn_cfg['d_model']}, n_layers={gnn_cfg['n_layers']}")

    print(f"\nServer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
