"""LaTeX tree parser and glyph counting utilities.

Provides recursive-descent parsing of our template LaTeX grammar into
an LNode tree, plus glyph/frac-bar counting used by tree data generation.
"""

from __future__ import annotations


# ── LaTeX tree parser ────────────────────────────────────────────────


class LNode:
    """Node in a parsed LaTeX tree."""

    __slots__ = ("kind", "children", "text", "start", "end")

    def __init__(
        self,
        kind: str,
        children: list[LNode] | None = None,
        text: str | None = None,
        start: int = 0,
        end: int = 0,
    ) -> None:
        self.kind = kind        # char, command, frac, sqrt, binom, sup, sub, func, seq
        self.children = children or []
        self.text = text        # for char/command: the character or command string
        self.start = start      # start position in LaTeX string
        self.end = end          # end position in LaTeX string


# Commands that render as a single character glyph
_SINGLE_GLYPH_CMDS = {
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon",
    "\\theta", "\\lambda", "\\mu", "\\pi", "\\sigma",
    "\\phi", "\\psi", "\\omega",
    "\\Gamma", "\\Delta", "\\Sigma", "\\Pi", "\\Phi", "\\Psi", "\\Omega",
    "\\sum", "\\int", "\\prod",
    "\\times", "\\cdot", "\\pm", "\\div",
    "\\leq", "\\geq", "\\neq",
    "\\in", "\\subset", "\\cup", "\\cap", "\\forall", "\\exists",
    "\\partial", "\\nabla", "\\infty",
    "\\rightarrow", "\\leftarrow",
    "\\ldots", "\\cdots",
    "\\EXPR",
}

# Function commands → number of rendered character glyphs
_FUNC_GLYPH_COUNTS = {
    "\\sin": 3, "\\cos": 3, "\\tan": 3,
    "\\log": 3, "\\ln": 2, "\\lim": 3,
}


def parse_latex(latex: str) -> LNode | None:
    """Parse LaTeX into a tree. Returns None on failure."""
    try:
        pos = [0]

        def peek():
            return latex[pos[0]] if pos[0] < len(latex) else ""

        def advance():
            pos[0] += 1

        def skip_ws():
            while pos[0] < len(latex) and latex[pos[0]].isspace():
                advance()

        def read_cmd():
            start = pos[0]
            advance()  # skip backslash
            if pos[0] < len(latex) and not latex[pos[0]].isalpha():
                advance()
                return latex[start : pos[0]]
            while pos[0] < len(latex) and latex[pos[0]].isalpha():
                advance()
            return latex[start : pos[0]]

        def parse_group():
            group_start = pos[0]
            advance()  # skip {
            inner = parse_expr()
            if peek() == "}":
                advance()
            # Extend inner span to include the braces so replacement
            # correctly removes {content} and substitutes {\\EXPR}
            inner.start = min(inner.start, group_start)
            inner.end = max(inner.end, pos[0])
            return inner

        def parse_primary():
            skip_ws()
            if pos[0] >= len(latex):
                return None

            start = pos[0]
            c = peek()

            if c == "{":
                return parse_group()

            if c == "\\":
                cmd = read_cmd()
                cmd_end = pos[0]

                if cmd in ("\\left", "\\right"):
                    # Skip \left/\right, parse the next token normally
                    return parse_primary()

                if cmd == "\\frac":
                    skip_ws()
                    num = parse_group()
                    skip_ws()
                    den = parse_group()
                    return LNode("frac", [num, den], start=start, end=pos[0])

                if cmd == "\\sqrt":
                    skip_ws()
                    content = parse_group()
                    return LNode("sqrt", [content], start=start, end=pos[0])

                if cmd == "\\binom":
                    skip_ws()
                    top = parse_group()
                    skip_ws()
                    bot = parse_group()
                    return LNode("binom", [top, bot], start=start, end=pos[0])

                if cmd in _FUNC_GLYPH_COUNTS:
                    skip_ws()
                    if peek() == "{":
                        arg = parse_group()
                        return LNode(
                            "func",
                            [LNode("command", text=cmd, start=start, end=cmd_end), arg],
                            text=cmd,
                            start=start,
                            end=pos[0],
                        )
                    return LNode("command", text=cmd, start=start, end=cmd_end)

                if cmd in _SINGLE_GLYPH_CMDS:
                    return LNode("command", text=cmd, start=start, end=cmd_end)

                # Unknown command — treat as single glyph
                return LNode("command", text=cmd, start=start, end=cmd_end)

            if c in ("^", "_", "}"):
                return None

            advance()
            return LNode("char", text=c, start=start, end=pos[0])

        def parse_term():
            base = parse_primary()
            if base is None:
                return None

            while True:
                skip_ws()
                c = peek()
                if c == "^":
                    advance()
                    skip_ws()
                    exp = parse_group() if peek() == "{" else parse_primary()
                    if exp is None:
                        break
                    base = LNode("sup", [base, exp], start=base.start, end=pos[0])
                elif c == "_":
                    advance()
                    skip_ws()
                    sub = parse_group() if peek() == "{" else parse_primary()
                    if sub is None:
                        break
                    base = LNode("sub", [base, sub], start=base.start, end=pos[0])
                else:
                    break

            return base

        def parse_expr():
            terms = []
            while pos[0] < len(latex) and peek() != "}":
                term = parse_term()
                if term is None:
                    break
                terms.append(term)

            if not terms:
                return LNode("seq", [], start=pos[0], end=pos[0])
            if len(terms) == 1:
                return terms[0]
            return LNode("seq", terms, start=terms[0].start, end=terms[-1].end)

        return parse_expr()
    except Exception:
        return None


# ── Glyph counting ──────────────────────────────────────────────────


def _n_char_glyphs(node: LNode) -> int:
    """Number of character glyphs (excluding frac bars) in this subtree."""
    if node.kind == "char":
        return 1
    if node.kind == "command":
        return _FUNC_GLYPH_COUNTS.get(node.text, 1)
    if node.kind == "sqrt":
        # sqrt radical = 1 glyph + children
        return 1 + sum(_n_char_glyphs(c) for c in node.children)
    if node.kind == "binom":
        # binom renders as ( children ) — 2 extra parens
        return 2 + sum(_n_char_glyphs(c) for c in node.children)
    return sum(_n_char_glyphs(c) for c in node.children)


def _n_frac_bars(node: LNode) -> int:
    """Number of fraction bars in this subtree."""
    count = 1 if node.kind == "frac" else 0
    return count + sum(_n_frac_bars(c) for c in node.children)


