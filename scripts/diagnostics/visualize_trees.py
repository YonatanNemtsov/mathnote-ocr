"""Generate an HTML page showing LaTeX expressions alongside their tree structures."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mathnote_ocr.latex_utils.glyphs import _extract_glyphs
from mathnote_ocr.tree_parser.gen_data import latex_to_tree_labels
from mathnote_ocr.tree_parser.tree import (
    DEN,
    EDGE_NAMES,
    LOWER,
    NUM,
    SQRT_CONTENT,
    SUB,
    SUP,
    UPPER,
    SymbolNode,
    build_tree,
    tree_to_latex,
)

EDGE_COLORS = {
    NUM: "#2196F3",
    DEN: "#F44336",
    SUP: "#4CAF50",
    SUB: "#FF9800",
    SQRT_CONTENT: "#9C27B0",
    UPPER: "#00BCD4",
    LOWER: "#795548",
}

EXAMPLES = [
    r"x^{2}+y^{2}=z^{2}",
    r"\frac{a+b}{c}",
    r"\frac{1}{\frac{x}{y}+1}",
    r"\frac{\frac{a}{b}}{\frac{c}{d}}",
    r"{\sum}_{i=0}^{n}x^{i}",
    r"\sqrt{a^{2}+b^{2}}",
    r"\sin{x}+\cos{y}",
    r"\binom{n}{k}",
    r"\frac{{\partial}f}{{\partial}x}",
    r"{\int}_{0}^{{\infty}}e^{-x}",
    r"{\lim}_{n{\rightarrow}{\infty}}\frac{1}{n}",
    r"\log_{2}{x}",
    r"\sin^{2}{x}+\cos^{2}{x}=1",
    r"\tan{\theta}",
]


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _dot_id(node: SymbolNode) -> str:
    return f"n{node.index}"


def _build_dot(roots: list[SymbolNode], glyph_names: list[str]) -> str:
    """Build a graphviz DOT string for the tree."""
    lines = [
        "digraph T {",
        "  rankdir=TB;",
        '  node [shape=box, style="rounded,filled", fillcolor="#f5f5f5",'
        '        fontname="Courier", fontsize=14, margin="0.15,0.07"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    # Collect all nodes via BFS
    all_nodes: list[SymbolNode] = []

    def _collect(node: SymbolNode):
        all_nodes.append(node)
        for et, children in sorted(node.children.items()):
            for c in children:
                _collect(c)

    for r in roots:
        _collect(r)

    # Add nodes
    for node in all_nodes:
        label = _escape(glyph_names[node.index])
        lines.append(f'  {_dot_id(node)} [label="{label}"];')

    # Add parent→child edges
    for node in all_nodes:
        for et, children in sorted(node.children.items()):
            color = EDGE_COLORS.get(et, "#000000")
            ename = EDGE_NAMES[et]
            for c in children:
                lines.append(
                    f"  {_dot_id(node)} -> {_dot_id(c)}"
                    f' [label=" {ename}", color="{color}",'
                    f'  fontcolor="{color}"];'
                )

    # Add seq (sibling) edges
    for node in all_nodes:
        for et, children in sorted(node.children.items()):
            for i in range(1, len(children)):
                prev = children[i - 1]
                curr = children[i]
                lines.append(
                    f"  {_dot_id(prev)} -> {_dot_id(curr)}"
                    f' [label=" seq", color="#999999", fontcolor="#999999",'
                    f"  style=dashed, constraint=false];"
                )
    # Also seq edges among roots
    for i in range(1, len(roots)):
        lines.append(
            f"  {_dot_id(roots[i - 1])} -> {_dot_id(roots[i])}"
            f' [label=" seq", color="#999999", fontcolor="#999999",'
            f"  style=dashed, constraint=false];"
        )

    lines.append("}")
    return "\n".join(lines)


def _dot_to_svg(dot_src: str) -> str:
    """Render DOT to SVG via graphviz."""
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot_src,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"<pre>graphviz error: {_escape(result.stderr)}</pre>"
    # Strip XML header, keep just the <svg> tag
    svg = result.stdout
    idx = svg.find("<svg")
    return svg[idx:] if idx >= 0 else svg


def process_example(latex: str) -> dict | None:
    """Process a LaTeX expression and return tree info."""
    glyphs = _extract_glyphs(latex)
    if glyphs is None:
        return None

    n_glyphs = len(glyphs)
    result = latex_to_tree_labels(latex, n_glyphs)
    if result is None:
        return None

    glyph_names = [g["name"] for g in glyphs]

    nodes = []
    for j, (name, (par, edge, order)) in enumerate(zip(glyph_names, result)):
        nodes.append(
            SymbolNode(
                symbol=name,
                bbox=[0, 0, 1, 1],
                index=j,
                parent=par,
                edge_type=edge,
                order=order,
            )
        )

    roots = build_tree(nodes)
    reconstructed = tree_to_latex(roots)
    dot_src = _build_dot(roots, glyph_names)
    svg = _dot_to_svg(dot_src)

    return {
        "latex": latex,
        "glyphs": glyph_names,
        "reconstructed": reconstructed,
        "svg": svg,
        "match": reconstructed == latex,
    }


def generate_html(examples: list[str], out_path: str):
    """Generate an HTML page with LaTeX + tree visualizations."""
    cards = []
    for latex in examples:
        info = process_example(latex)
        if info is None:
            cards.append(
                f'<div class="card"><p>Failed to process: <code>{_escape(latex)}</code></p></div>'
            )
            continue

        match_icon = "&#10003;" if info["match"] else "&#10007;"
        match_class = "match" if info["match"] else "nomatch"

        cards.append(f"""
        <div class="card">
          <div class="latex-side">
            <div class="section-label">LaTeX</div>
            <div class="rendered">$${_escape(info["latex"])}$$</div>
            <div class="code"><code>{_escape(info["latex"])}</code></div>
            <div class="glyphs">Glyphs: {", ".join(info["glyphs"])}</div>
            <div class="roundtrip {match_class}">
              <span>{match_icon}</span>
              <code>{_escape(info["reconstructed"])}</code>
            </div>
          </div>
          <div class="tree-side">
            <div class="section-label">Tree</div>
            {info["svg"]}
          </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Math Expression Trees</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; background: #fafafa; margin: 20px; }}
  h1 {{ text-align: center; color: #333; }}
  .legend {{
    display: flex; gap: 18px; justify-content: center;
    margin: 10px 0 25px; font-size: 13px;
  }}
  .legend span {{
    display: inline-flex; align-items: center; gap: 4px;
  }}
  .legend .dot {{
    width: 12px; height: 12px; border-radius: 50%; display: inline-block;
  }}
  .card {{
    display: flex; gap: 30px; align-items: flex-start;
    background: white; border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    padding: 24px; margin-bottom: 20px;
  }}
  .latex-side {{ flex: 0 0 340px; }}
  .tree-side {{ flex: 1; overflow-x: auto; }}
  .tree-side svg {{ max-width: 100%; height: auto; }}
  .section-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    color: #999; margin-bottom: 8px;
  }}
  .rendered {{ font-size: 22px; margin-bottom: 10px; }}
  .code {{ font-size: 12px; color: #555; word-break: break-all; margin-bottom: 6px; }}
  .glyphs {{ font-size: 11px; color: #888; margin-bottom: 6px; }}
  .roundtrip {{ font-size: 12px; }}
  .roundtrip.match {{ color: #4CAF50; }}
  .roundtrip.nomatch {{ color: #F44336; }}
  .roundtrip code {{ color: inherit; }}
</style>
</head>
<body>
<h1>Math Expression Trees</h1>
<div class="legend">
  <span><span class="dot" style="background:{EDGE_COLORS[NUM]}"></span> num</span>
  <span><span class="dot" style="background:{EDGE_COLORS[DEN]}"></span> den</span>
  <span><span class="dot" style="background:{EDGE_COLORS[SUP]}"></span> sup</span>
  <span><span class="dot" style="background:{EDGE_COLORS[SUB]}"></span> sub</span>
  <span><span class="dot" style="background:{EDGE_COLORS[SQRT_CONTENT]}"></span> sqrt</span>
  <span><span class="dot" style="background:{EDGE_COLORS[UPPER]}"></span> upper</span>
  <span><span class="dot" style="background:{EDGE_COLORS[LOWER]}"></span> lower</span>
</div>
{"".join(cards)}
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "tree_vis.html")
    generate_html(EXAMPLES, out)
