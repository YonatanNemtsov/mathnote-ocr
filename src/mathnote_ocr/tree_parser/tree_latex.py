"""Clean tree ↔ LaTeX conversion.

LaTeX conventions:
- Commands separated by spaces: \\alpha x + \\beta y
- Braces only for ^{} _{} and structural commands (\\frac, \\sqrt, \\binom)
- No unnecessary {\\cmd} wrapping
- Big ops: \\int_{a}^{b}, \\sum_{i=0}^{n} (no outer braces)
- Functions: \\sin, \\cos etc. rendered as commands (not individual letters)

tree_to_latex: Tree → clean LaTeX string
latex_to_tree: clean LaTeX string → Tree
"""

from __future__ import annotations

from mathnote_ocr.bbox import BBox
from mathnote_ocr.tree_parser.tree_v2 import ROOT_ID, Edge, Node, Symbol, SymbolId, Tree

# ── Symbol name ↔ LaTeX command ─────────────────────────────────────

# Symbol names that map to LaTeX commands (need backslash)
_NAME_TO_CMD: dict[str, str] = {
    # Greek lowercase
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "theta": r"\theta",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "pi": r"\pi",
    "sigma": r"\sigma",
    "phi": r"\phi",
    "psi": r"\psi",
    "omega": r"\omega",
    # Greek uppercase
    "Gamma_cap": r"\Gamma",
    "Delta_cap": r"\Delta",
    "Sigma_up": r"\Sigma",
    "Pi_up": r"\Pi",
    "Phi_up": r"\Phi",
    "Psi_up": r"\Psi",
    "Omega_up": r"\Omega",
    # Big operators
    "int": r"\int",
    "sum": r"\sum",
    "prod": r"\prod",
    # Binary operators
    "times": r"\times",
    "cdot": r"\cdot",
    "pm": r"\pm",
    "div": r"\div",
    # Relations
    "leq": r"\leq",
    "geq": r"\geq",
    "neq": r"\neq",
    # Set / logic
    "in": r"\in",
    "subset": r"\subset",
    "cup": r"\cup",
    "cap": r"\cap",
    "forall": r"\forall",
    "exists": r"\exists",
    # Calculus / special
    "partial": r"\partial",
    "nabla": r"\nabla",
    "infty": r"\infty",
    "rightarrow": r"\rightarrow",
    "leftarrow": r"\leftarrow",
    "ldots": r"\ldots",
    "cdots": r"\cdots",
    # Delimiters
    "lbrace": r"\lbrace",
    "rbrace": r"\rbrace",
    # Prime
    "prime": r"\prime",
    # Capitals: A_cap → A etc.
    **{f"{c}_cap": c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
}

# Reverse: LaTeX command → symbol name
_CMD_TO_NAME: dict[str, str] = {v: k for k, v in _NAME_TO_CMD.items()}

# Special cases
_SPECIAL_NAMES = {"dot": ".", "frac_bar": "-", "slash": "/", "colon": ":"}
_SPECIAL_CHARS = {v: k for k, v in _SPECIAL_NAMES.items()}

# Big ops and functions (for rendering logic)
_BIG_OPS = {"int", "sum", "prod"}
_FUNC_SEQUENCES = {
    ("s", "i", "n"): r"\sin",
    ("c", "o", "s"): r"\cos",
    ("t", "a", "n"): r"\tan",
    ("l", "o", "g"): r"\log",
    ("l", "n"): r"\ln",
    ("l", "i", "m"): r"\lim",
}


_OPEN_TO_CLOSE = {"(": ")", "[": "]", "lbrace": "rbrace"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}
_LEFT_PREFIX = {
    "(": r"\left(",
    ")": r"\right)",
    "[": r"\left[",
    "]": r"\right]",
    "lbrace": r"\left\lbrace",
    "rbrace": r"\right\rbrace",
}


def _sym_to_latex(name: str) -> str:
    """Convert symbol name to LaTeX token (no braces)."""
    if name in _NAME_TO_CMD:
        return _NAME_TO_CMD[name]
    if name in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[name]
    return name


# ── Tree → LaTeX ────────────────────────────────────────────────────


def tree_to_latex(tree: Tree) -> str:
    """Convert a tree to a clean LaTeX string."""
    return _render_siblings(tree, tree.root_ids())


def _find_matched_parens(tree: Tree, ids: tuple[SymbolId, ...]) -> set[int]:
    """Find indices into ids that have a matching open/close partner.
    Returns set of indices that should use \\left/\\right."""
    matched: set[int] = set()
    stack: list[tuple[str, int]] = []  # (open_name, index)
    for i, sid in enumerate(ids):
        name = tree[sid].symbol.name
        if name in _OPEN_TO_CLOSE:
            stack.append((name, i))
        elif name in _CLOSE_TO_OPEN:
            expected_open = _CLOSE_TO_OPEN[name]
            # Find matching open in stack (search from top)
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == expected_open:
                    matched.add(stack[j][1])
                    matched.add(i)
                    stack.pop(j)
                    break
    return matched


def _render_siblings(tree: Tree, ids: tuple[SymbolId, ...]) -> str:
    matched = _find_matched_parens(tree, ids)

    parts: list[str] = []
    i = 0
    while i < len(ids):
        # Try to match function sequences (s,i,n → \sin)
        match = _try_match_func(tree, ids, i)
        if match is not None:
            cmd, n_glyphs = match
            tail = ids[i + n_glyphs - 1]
            rendered = _render_limits(tree, tail, cmd)
            rendered = _render_sup_sub(tree, tail, rendered)
            parts.append(rendered)
            i += n_glyphs
        else:
            name = tree[ids[i]].symbol.name
            if i in matched and name in _LEFT_PREFIX:
                rendered = _LEFT_PREFIX[name]
                rendered = _render_sup_sub(tree, ids[i], rendered)
                parts.append(rendered)
            else:
                parts.append(_render_node(tree, ids[i]))
            i += 1

    return " ".join(parts)


def _render_sup_sub(tree: Tree, sid: SymbolId, base: str) -> str:
    result = base
    has_sub = _has_kids(tree, sid, Edge.SUB)
    has_sup = _has_kids(tree, sid, Edge.SUP)
    if has_sub and has_sup:
        # Render in order of first child ID (preserves parse order)
        sub_first_id = _kids(tree, sid, Edge.SUB)[0]
        sup_first_id = _kids(tree, sid, Edge.SUP)[0]
        if sub_first_id < sup_first_id:
            result += f"_{{{_render_siblings(tree, _kids(tree, sid, Edge.SUB))}}}"
            result += f"^{{{_render_siblings(tree, _kids(tree, sid, Edge.SUP))}}}"
        else:
            result += f"^{{{_render_siblings(tree, _kids(tree, sid, Edge.SUP))}}}"
            result += f"_{{{_render_siblings(tree, _kids(tree, sid, Edge.SUB))}}}"
    elif has_sub:
        result += f"_{{{_render_siblings(tree, _kids(tree, sid, Edge.SUB))}}}"
    elif has_sup:
        result += f"^{{{_render_siblings(tree, _kids(tree, sid, Edge.SUP))}}}"
    return result


def _render_limits(tree: Tree, sid: SymbolId, base: str) -> str:
    """Render LOWER/UPPER limits (for big ops and functions)."""
    result = base
    if _has_kids(tree, sid, Edge.LOWER):
        result += f"_{{{_render_siblings(tree, _kids(tree, sid, Edge.LOWER))}}}"
    if _has_kids(tree, sid, Edge.UPPER):
        result += f"^{{{_render_siblings(tree, _kids(tree, sid, Edge.UPPER))}}}"
    return result


def _render_node(tree: Tree, sid: SymbolId) -> str:
    name = tree[sid].symbol.name

    # Binom: ( with NUM, DEN, MATCH children
    if (
        name == "("
        and _has_kids(tree, sid, Edge.NUM)
        and _has_kids(tree, sid, Edge.DEN)
        and _has_kids(tree, sid, Edge.MATCH)
    ):
        num = _render_siblings(tree, _kids(tree, sid, Edge.NUM))
        den = _render_siblings(tree, _kids(tree, sid, Edge.DEN))
        result = f"\\binom{{{num}}}{{{den}}}"
        for m in _kids(tree, sid, Edge.MATCH):
            result = _render_sup_sub(tree, m, result)
        return result

    # Fraction bar (complete or partial)
    if name in ("-", "frac_bar") and (
        _has_kids(tree, sid, Edge.NUM) or _has_kids(tree, sid, Edge.DEN)
    ):
        num = (
            _render_siblings(tree, _kids(tree, sid, Edge.NUM))
            if _has_kids(tree, sid, Edge.NUM)
            else ""
        )
        den = (
            _render_siblings(tree, _kids(tree, sid, Edge.DEN))
            if _has_kids(tree, sid, Edge.DEN)
            else ""
        )
        return _render_sup_sub(tree, sid, f"\\frac{{{num}}}{{{den}}}")

    # Sqrt
    if name == "sqrt" and _has_kids(tree, sid, Edge.SQRT):
        content = _render_siblings(tree, _kids(tree, sid, Edge.SQRT))
        return _render_sup_sub(tree, sid, f"\\sqrt{{{content}}}")

    # Big operator
    if name in _BIG_OPS:
        latex = _sym_to_latex(name)
        latex = _render_limits(tree, sid, latex)
        return _render_sup_sub(tree, sid, latex)

    # Symbol with limits (shouldn't normally happen outside big ops/funcs)
    if _has_kids(tree, sid, Edge.LOWER) or _has_kids(tree, sid, Edge.UPPER):
        latex = _sym_to_latex(name)
        latex = _render_limits(tree, sid, latex)
        return _render_sup_sub(tree, sid, latex)

    # Regular symbol
    return _render_sup_sub(tree, sid, _sym_to_latex(name))


def _has_kids(tree: Tree, sid: SymbolId, edge: int) -> bool:
    return len(_kids(tree, sid, edge)) > 0


def _kids(tree: Tree, sid: SymbolId, edge: int) -> tuple[SymbolId, ...]:
    return tree.children_by_edge(sid, edge)


def _try_match_func(tree: Tree, ids: tuple[SymbolId, ...], start: int) -> tuple[str, int] | None:
    if start >= len(ids):
        return None
    for seq, cmd in _FUNC_SEQUENCES.items():
        end = start + len(seq)
        if end > len(ids):
            continue
        if all(
            tree[ids[start + k]].symbol.name == seq[k]
            or (seq[k] == "o" and tree[ids[start + k]].symbol.name == "0")
            for k in range(len(seq))
        ):
            return cmd, len(seq)
    return None


# ── LaTeX → Tree ────────────────────────────────────────────────────


def latex_to_tree(latex: str) -> Tree:
    """Parse clean LaTeX into a Tree.

    Handles: \\frac{}{}, \\sqrt{}, \\binom{}{}, ^{}, _{},
    \\cmd tokens, single chars, spaces as separators.
    """
    tokens = _tokenize(latex)
    nodes, _ = _parse_expr(tokens, 0)
    return _nodes_to_tree(nodes, ROOT_ID, Edge.ROOT)


def _tokenize(latex: str) -> list[str]:
    """Split LaTeX into tokens: commands, single chars, braces, ^, _."""
    tokens = []
    i = 0
    while i < len(latex):
        c = latex[i]
        if c.isspace():
            i += 1
            continue
        if c == "\\":
            # Read command
            j = i + 1
            if j < len(latex) and not latex[j].isalpha():
                tokens.append(latex[i : j + 1])
                i = j + 1
            else:
                while j < len(latex) and latex[j].isalpha():
                    j += 1
                tokens.append(latex[i:j])
                i = j
        elif c in "{}_^":
            tokens.append(c)
            i += 1
        else:
            tokens.append(c)
            i += 1
    return tokens


class _ParseNode:
    """Intermediate parse node before converting to Tree."""

    __slots__ = ("name", "children")

    def __init__(self, name: str):
        self.name = name
        # children: dict of edge_type → list[_ParseNode]
        self.children: dict[int, list[_ParseNode]] = {}

    def add_child(self, edge: int, child: _ParseNode):
        self.children.setdefault(edge, []).append(child)


def _cmd_to_name(cmd: str) -> str:
    """Convert a LaTeX command to our symbol name."""
    if cmd in _CMD_TO_NAME:
        return _CMD_TO_NAME[cmd]
    # Strip backslash for unknown commands
    return cmd.lstrip("\\")


def _char_to_name(ch: str) -> str:
    """Convert a single character to our symbol name."""
    if ch in _SPECIAL_CHARS:
        return _SPECIAL_CHARS[ch]
    # Uppercase letters → X_cap
    if ch.isupper():
        return f"{ch}_cap"
    return ch


def _parse_group(tokens: list[str], pos: int) -> tuple[list[_ParseNode], int]:
    """Parse {...} group, returns (nodes, new_pos). Expects pos at '{'."""
    assert tokens[pos] == "{", f"Expected '{{' at pos {pos}, got '{tokens[pos]}'"
    pos += 1  # skip {
    nodes, pos = _parse_expr(tokens, pos)
    if pos < len(tokens) and tokens[pos] == "}":
        pos += 1  # skip }
    return nodes, pos


def _parse_expr(tokens: list[str], pos: int) -> tuple[list[_ParseNode], int]:
    """Parse a sequence of terms until } or end."""
    nodes: list[_ParseNode] = []
    while pos < len(tokens) and tokens[pos] != "}":
        result, pos = _parse_term(tokens, pos)
        if result is None:
            continue
        if isinstance(result, list):
            nodes.extend(result)
        else:
            nodes.append(result)
    return nodes, pos


def _parse_term(tokens: list[str], pos: int) -> tuple[_ParseNode | None, int]:
    """Parse one term: a base with optional ^{} and _{}."""
    if pos >= len(tokens) or tokens[pos] == "}":
        return None, pos

    tok = tokens[pos]

    # \left and \right — strip, parse the next token as the delimiter
    if tok in (r"\left", r"\right"):
        pos += 1
        return _parse_term(tokens, pos)

    # \frac{num}{den}
    if tok == r"\frac":
        pos += 1
        num_nodes, pos = _parse_group(tokens, pos)
        den_nodes, pos = _parse_group(tokens, pos)
        node = _ParseNode("frac_bar")
        for n in num_nodes:
            node.add_child(Edge.NUM, n)
        for n in den_nodes:
            node.add_child(Edge.DEN, n)
        pos = _parse_sup_sub(tokens, pos, node)
        return node, pos

    # \sqrt{content}
    if tok == r"\sqrt":
        pos += 1
        content_nodes, pos = _parse_group(tokens, pos)
        node = _ParseNode("sqrt")
        for n in content_nodes:
            node.add_child(Edge.SQRT, n)
        pos = _parse_sup_sub(tokens, pos, node)
        return node, pos

    # \binom{top}{bot}
    if tok == r"\binom":
        pos += 1
        top_nodes, pos = _parse_group(tokens, pos)
        bot_nodes, pos = _parse_group(tokens, pos)
        node = _ParseNode("(")
        for n in top_nodes:
            node.add_child(Edge.NUM, n)
        for n in bot_nodes:
            node.add_child(Edge.DEN, n)
        # Add closing paren as MATCH
        close = _ParseNode(")")
        node.add_child(Edge.MATCH, close)
        pos = _parse_sup_sub(tokens, pos, node)
        return node, pos

    # Bare { group — just parse contents inline
    if tok == "{":
        inner_nodes, pos = _parse_group(tokens, pos)
        # If single node, attach sup/sub to it
        if len(inner_nodes) == 1:
            pos = _parse_sup_sub(tokens, pos, inner_nodes[0])
            return inner_nodes[0], pos
        # Multiple nodes — return them as-is (caller handles siblings)
        # But we need to handle sup/sub on the last node
        if inner_nodes:
            pos = _parse_sup_sub(tokens, pos, inner_nodes[-1])
        # Return all nodes — we'll flatten in the caller
        return inner_nodes, pos  # type: ignore

    # Function command → expand to individual letter nodes
    _FUNC_CMD_TO_LETTERS = {
        r"\sin": list("sin"),
        r"\cos": list("cos"),
        r"\tan": list("tan"),
        r"\log": list("log"),
        r"\ln": list("ln"),
        r"\lim": list("lim"),
    }
    # Functions that use LOWER/UPPER limits instead of SUB/SUP
    _FUNC_WITH_LIMITS = {r"\lim"}
    if tok in _FUNC_CMD_TO_LETTERS:
        letters = _FUNC_CMD_TO_LETTERS[tok]
        nodes = [_ParseNode(ch) for ch in letters]
        pos += 1
        pos = _parse_sup_sub(tokens, pos, nodes[-1], use_limits=(tok in _FUNC_WITH_LIMITS))
        return nodes, pos  # type: ignore

    # Command token
    if tok.startswith("\\"):
        name = _cmd_to_name(tok)
        node = _ParseNode(name)
        pos += 1
        pos = _parse_sup_sub(tokens, pos, node)
        return node, pos

    # Single character
    name = _char_to_name(tok)
    node = _ParseNode(name)
    pos += 1
    pos = _parse_sup_sub(tokens, pos, node)
    return node, pos


def _parse_sup_sub(tokens: list[str], pos: int, node: _ParseNode, use_limits: bool = False) -> int:
    """Parse optional ^{} and _{} after a node.

    If use_limits, uses UPPER/LOWER edges instead of SUP/SUB.
    """
    if not use_limits:
        use_limits = node.name in _BIG_OPS
    up_edge = Edge.UPPER if use_limits else Edge.SUP
    down_edge = Edge.LOWER if use_limits else Edge.SUB

    while pos < len(tokens) and tokens[pos] in ("^", "_"):
        if tokens[pos] == "^":
            pos += 1
            if pos < len(tokens) and tokens[pos] == "{":
                children, pos = _parse_group(tokens, pos)
                for c in children:
                    node.add_child(up_edge, c)
            elif pos < len(tokens):
                child, pos = _parse_term(tokens, pos)
                if child is not None:
                    node.add_child(up_edge, child)
        elif tokens[pos] == "_":
            pos += 1
            if pos < len(tokens) and tokens[pos] == "{":
                children, pos = _parse_group(tokens, pos)
                for c in children:
                    node.add_child(down_edge, c)
            elif pos < len(tokens):
                child, pos = _parse_term(tokens, pos)
                if child is not None:
                    node.add_child(down_edge, child)
    return pos


def _nodes_to_tree(
    nodes: list[_ParseNode],
    parent_id: SymbolId,
    edge_type: int,
) -> Tree:
    """Convert list of _ParseNode into a Tree, assigning IDs depth-first."""
    all_nodes: list[Node] = []
    counter = [0]

    def _walk(pnodes: list[_ParseNode], par_id: SymbolId, edge: int):
        for order, pnode in enumerate(pnodes):
            sid = counter[0]
            counter[0] += 1
            sym = Symbol(sid, pnode.name, BBox(0, 0, 0, 0))
            all_nodes.append(Node(sym, par_id, edge, order))
            # Recurse into children by edge type
            for child_edge, child_pnodes in pnode.children.items():
                _walk(child_pnodes, sid, child_edge)

    _walk(nodes, parent_id, edge_type)
    return Tree(tuple(all_nodes))
