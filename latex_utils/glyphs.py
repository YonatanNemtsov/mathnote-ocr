"""Unicode-to-symbol mapping and glyph extraction from LaTeX via ziamath."""

import re

import ziamath as zm


# ── Unicode to symbol name mapping ───────────────────────────────────

CHAR_TO_SYMBOL: dict[int, str] = {}
# Digits
for c in "0123456789":
    CHAR_TO_SYMBOL[ord(c)] = c
# Lowercase
for c in "abcdefghijklmnopqrstuvwxyz":
    CHAR_TO_SYMBOL[ord(c)] = c
# Uppercase
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    CHAR_TO_SYMBOL[ord(c)] = f"{c}_cap"
# Operators
CHAR_TO_SYMBOL[ord("+")] = "+"
CHAR_TO_SYMBOL[0x2212] = "-"       # Unicode minus
CHAR_TO_SYMBOL[ord("-")] = "-"     # ASCII minus
CHAR_TO_SYMBOL[ord("=")] = "="
CHAR_TO_SYMBOL[0x00D7] = "times"   # ×
CHAR_TO_SYMBOL[0x22C5] = "cdot"    # ⋅ (dot operator)
CHAR_TO_SYMBOL[0x00B7] = "cdot"    # · (middle dot, ziamath's \cdot)
CHAR_TO_SYMBOL[ord(".")] = "dot"   # decimal point
# Parens / brackets
CHAR_TO_SYMBOL[ord("(")] = "("
CHAR_TO_SYMBOL[ord(")")] = ")"
CHAR_TO_SYMBOL[ord("[")] = "["
CHAR_TO_SYMBOL[ord("]")] = "]"
CHAR_TO_SYMBOL[ord("{")] = "lbrace"
CHAR_TO_SYMBOL[ord("}")] = "rbrace"
# Punctuation
CHAR_TO_SYMBOL[ord("!")] = "!"
CHAR_TO_SYMBOL[ord(",")] = ","
CHAR_TO_SYMBOL[ord(";")] = ";"
CHAR_TO_SYMBOL[ord(":")] = "colon"
CHAR_TO_SYMBOL[ord("|")] = "|"
CHAR_TO_SYMBOL[ord("/")] = "slash"
CHAR_TO_SYMBOL[ord("<")] = "<"
CHAR_TO_SYMBOL[ord(">")] = ">"
CHAR_TO_SYMBOL[0x2032] = "prime"   # ′
CHAR_TO_SYMBOL[ord("'")] = "prime" # ASCII apostrophe fallback

# Greek lowercase
CHAR_TO_SYMBOL[945] = "alpha"      # α
CHAR_TO_SYMBOL[946] = "beta"       # β
CHAR_TO_SYMBOL[947] = "gamma"      # γ
CHAR_TO_SYMBOL[948] = "delta"      # δ
CHAR_TO_SYMBOL[949] = "epsilon"    # ε
CHAR_TO_SYMBOL[0x03F5] = "epsilon" # ϵ (lunate epsilon, ziamath's \epsilon)
CHAR_TO_SYMBOL[952] = "theta"      # θ
CHAR_TO_SYMBOL[955] = "lambda"     # λ
CHAR_TO_SYMBOL[956] = "mu"         # μ
CHAR_TO_SYMBOL[960] = "pi"         # π
CHAR_TO_SYMBOL[963] = "sigma"      # σ
CHAR_TO_SYMBOL[981] = "phi"        # ϕ
CHAR_TO_SYMBOL[968] = "psi"        # ψ
CHAR_TO_SYMBOL[969] = "omega"      # ω

# Greek uppercase
CHAR_TO_SYMBOL[915] = "Gamma_cap"   # Γ
CHAR_TO_SYMBOL[916] = "Delta_cap"  # Δ
CHAR_TO_SYMBOL[931] = "Sigma_up"   # Σ
CHAR_TO_SYMBOL[928] = "Pi_up"      # Π
CHAR_TO_SYMBOL[934] = "Phi_up"     # Φ
CHAR_TO_SYMBOL[936] = "Psi_up"     # Ψ
CHAR_TO_SYMBOL[937] = "Omega_up"   # Ω

# Big operators
CHAR_TO_SYMBOL[8721] = "sum"       # ∑ (distinct from Σ = Sigma_up)
CHAR_TO_SYMBOL[8747] = "int"       # ∫
CHAR_TO_SYMBOL[8719] = "prod"      # ∏ (distinct from Π = Pi_up)

# Comparisons / relations
CHAR_TO_SYMBOL[0x00B1] = "pm"      # ±
CHAR_TO_SYMBOL[0x00F7] = "div"     # ÷
CHAR_TO_SYMBOL[8804] = "leq"       # ≤
CHAR_TO_SYMBOL[8805] = "geq"       # ≥
CHAR_TO_SYMBOL[8800] = "neq"       # ≠

# Set / logic
CHAR_TO_SYMBOL[8712] = "in"        # ∈
CHAR_TO_SYMBOL[8834] = "subset"    # ⊂
CHAR_TO_SYMBOL[8746] = "cup"       # ∪
CHAR_TO_SYMBOL[8745] = "cap"       # ∩
CHAR_TO_SYMBOL[8704] = "forall"    # ∀
CHAR_TO_SYMBOL[8707] = "exists"    # ∃

# Calculus
CHAR_TO_SYMBOL[8706] = "partial"   # ∂
CHAR_TO_SYMBOL[8711] = "nabla"     # ∇

# Special
CHAR_TO_SYMBOL[8734] = "infty"     # ∞
CHAR_TO_SYMBOL[8730] = "sqrt"      # √
CHAR_TO_SYMBOL[8230] = "ldots"     # …

# Arrows
CHAR_TO_SYMBOL[8594] = "rightarrow"  # →
CHAR_TO_SYMBOL[8592] = "leftarrow"   # ←


# ── Glyph extraction (ziamath, display-style) ───────────────────────


def _collect_chars(node, chars=None):
    """Collect visible glyph characters from node tree (matches SVG <use> order)."""
    if chars is None:
        chars = []
    if type(node).__name__ == "Glyph":
        if ord(node.char) > 32:
            chars.append(node.char)
    if hasattr(node, "nodes"):
        for child in node.nodes:
            _collect_chars(child, chars)
    return chars


def _collect_hline_parents(node, parents=None, parent_kind=""):
    """Collect parent types of HLine nodes (matches SVG <rect> order)."""
    if parents is None:
        parents = []
    kind = type(node).__name__
    if kind == "HLine":
        parents.append(parent_kind)
    elif hasattr(node, "nodes"):
        for child in node.nodes:
            _collect_hline_parents(child, parents, kind)
    return parents


_NUM_RE = re.compile(r"[-+]?[\d]*\.?[\d]+")
_SVG_NS = "http://www.w3.org/2000/svg"


def _ink_bbox(use_el, root, ns):
    """Extract ink bounding box from SVG <use> + <symbol> path data."""
    href = (use_el.get("href") or "").lstrip("#")
    ux = float(use_el.get("x", 0))
    uy = float(use_el.get("y", 0))
    uw = float(use_el.get("width", 0))
    uh = float(use_el.get("height", 0))

    for sym in root.iter(f"{{{ns}}}symbol"):
        if sym.get("id") == href:
            svb = [float(v) for v in sym.get("viewBox", "0 0 0 0").split()]
            sx, sy, sw, sh = svb
            path = sym.find(f"{{{ns}}}path")
            if path is None:
                return None
            d = path.get("d", "")
            nums = _NUM_RE.findall(d)
            coords = [float(n) for n in nums]
            xs, ys = coords[0::2], coords[1::2]
            scx, scy = uw / sw, uh / sh
            return (
                ux + (min(xs) - sx) * scx,
                uy + (min(ys) - sy) * scy,
                (max(xs) - min(xs)) * scx,
                (max(ys) - min(ys)) * scy,
            )
    return None


def _extract_glyphs(latex: str) -> list[dict] | None:
    """Parse LaTeX with ziamath (display-style) and extract per-glyph bboxes.

    Uses SVG output for screen-space coordinates and the node tree for
    character identity and HLine parent types.

    Returns list of {name, bbox: [x, y, w, h]} in screen coords (y-down),
    or None if parsing fails.
    """
    try:
        m = zm.Math.fromlatex(latex)
    except Exception:
        return None

    import xml.etree.ElementTree as ET

    svg_str = m.svg()
    root = ET.fromstring(svg_str)
    ns = _SVG_NS

    # Labels from node tree (1:1 match with SVG <use> elements)
    chars = _collect_chars(m.node)
    uses = list(root.iter(f"{{{ns}}}use"))
    if len(chars) != len(uses):
        return None

    # HLine parent types from node tree (1:1 match with SVG <rect> elements)
    hline_parents = _collect_hline_parents(m.node)
    svg_rects = list(root.iter(f"{{{ns}}}rect"))

    # Extract glyph ink bboxes from SVG, skip whitespace glyphs
    named: list[dict] = []
    for ch, use_el in zip(chars, uses):
        bb = _ink_bbox(use_el, root, ns)
        if bb is None:
            # Skip whitespace/invisible glyphs (e.g. thin space U+2009)
            if ord(ch) in (0x2009, 0x200A, 0x200B, 0x00A0):
                continue
            return None
        x, y, w, h = bb
        if w < 0.01:
            continue
        name = CHAR_TO_SYMBOL.get(ord(ch))
        if name is None:
            return None
        named.append({"name": name, "x": x, "y": y, "w": w, "h": h})

    # Process <rect> elements: merge sqrt overlines, keep frac bars
    for i, rect_el in enumerate(svg_rects):
        rx = float(rect_el.get("x", 0))
        ry = float(rect_el.get("y", 0))
        rw = float(rect_el.get("width", 0))
        rh = float(rect_el.get("height", 0))

        if rh < 0.01:
            continue  # invisible bar (e.g. \binom)

        parent = hline_parents[i] if i < len(hline_parents) else ""
        if parent == "Msqrt":
            # Merge with the nearest unmerged sqrt radical glyph
            best = None
            best_dist = float("inf")
            for g in named:
                if g["name"] == "sqrt" and not g.get("_sqrt_merged"):
                    dist = abs(rx - (g["x"] + g["w"]))
                    if dist < best_dist:
                        best = g
                        best_dist = dist
            if best is not None:
                best["_sqrt_merged"] = True
                nx = min(best["x"], rx)
                ny = min(best["y"], ry)
                nx2 = max(best["x"] + best["w"], rx + rw)
                ny2 = max(best["y"] + best["h"], ry + rh)
                best["x"], best["y"] = nx, ny
                best["w"], best["h"] = nx2 - nx, ny2 - ny
        else:
            named.append({"name": "frac_bar", "x": rx, "y": ry, "w": rw, "h": rh,
                          "is_frac_bar": True})

    if not named:
        return None

    # Find expression extent
    all_xmin = min(g["x"] for g in named)
    all_ymin = min(g["y"] for g in named)
    all_xmax = max(g["x"] + g["w"] for g in named)
    all_ymax = max(g["y"] + g["h"] for g in named)

    expr_w = all_xmax - all_xmin
    expr_h = all_ymax - all_ymin
    ref = max(expr_w, expr_h, 1.0)

    # Normalize to [0,1] range (already y-down from SVG)
    glyphs = []
    for g in named:
        entry = {
            "name": g["name"],
            "bbox": [
                round((g["x"] - all_xmin) / ref, 6),
                round((g["y"] - all_ymin) / ref, 6),
                round(g["w"] / ref, 6),
                round(g["h"] / ref, 6),
            ],
        }
        if g.get("is_frac_bar"):
            entry["is_frac_bar"] = True
        glyphs.append(entry)

    return glyphs


# ── Symbol name → LaTeX mapping ─────────────────────────────────────

SYMBOL_TO_LATEX: dict[str, str] = {
    # Capitals: A_cap → A, etc.
    **{f"{c}_cap": c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    # Big operators
    "int": r"\int", "sum": r"\sum", "prod": r"\prod",
    # Greek uppercase (Σ and Π are distinct from ∑ and ∏)
    "Sigma_up": r"\Sigma", "Pi_up": r"\Pi",
    # Greek lowercase
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "delta": r"\delta", "epsilon": r"\epsilon", "theta": r"\theta",
    "lambda": r"\lambda", "mu": r"\mu", "pi": r"\pi",
    "sigma": r"\sigma", "phi": r"\phi", "psi": r"\psi", "omega": r"\omega",
    # Greek uppercase
    "Gamma_cap": r"\Gamma", "Delta_cap": r"\Delta", "Phi_up": r"\Phi",
    "Psi_up": r"\Psi", "Omega_up": r"\Omega",
    # Binary operators
    "times": r"\times", "dot": ".", "cdot": r"\cdot", "pm": r"\pm", "div": r"\div",
    # Relations
    "leq": r"\leq", "geq": r"\geq", "neq": r"\neq",
    # Set / logic
    "in": r"\in", "subset": r"\subset", "cup": r"\cup", "cap": r"\cap",
    "forall": r"\forall", "exists": r"\exists",
    # Calculus / special
    "partial": r"\partial", "nabla": r"\nabla", "infty": r"\infty",
    "rightarrow": r"\rightarrow", "leftarrow": r"\leftarrow",
    "ldots": r"\ldots", "cdots": r"\cdots",
    # Trig / functions
    "sin": r"\sin", "cos": r"\cos", "tan": r"\tan",
    "log": r"\log", "ln": r"\ln", "lim": r"\lim",
    # Structural
    "frac_bar": "-",
    # Delimiters
    "lbrace": r"\lbrace", "rbrace": r"\rbrace",
    # Punctuation
    "slash": "/", "colon": ":",
    # Prime
    "prime": r"\prime",
}
