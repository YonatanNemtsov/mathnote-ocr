"""Augment font-rendered bboxes to look like handwritten bboxes (v2).

Perturbation-based: starts from font layout (correct structure),
applies per-symbol size corrections (from HW stats), and adds jitter.
"""

import random
from collections import defaultdict

from mathnote_ocr.tree_parser.tree import NUM, DEN, SUP, SUB, SQRT_CONTENT, UPPER, LOWER, MATCH


# ── Per-symbol size corrections ──────────────────────────────────────
# Measured from 275 matched HW expressions (run_001/002/003).
# (h_scale, h_std, w_scale, w_std) where scale = median(hw/font) ratio.
# Apply: new_h = font_h * gauss(h_scale, h_std), clamped.
_HW_SYMBOL_STATS = {
    '(': (1.126, 0.289, 0.868, 0.14),
    ')': (1.083, 0.265, 0.835, 0.172),
    '+': (0.891, 0.191, 0.798, 0.169),
    ',': (1.584, 0.297, 1.885, 0.253),
    '-': (2.636, 0.653, 0.683, 0.176),
    '0': (0.862, 0.245, 0.684, 0.186),
    '1': (0.818, 0.228, 0.785, 0.21),
    '2': (0.837, 0.309, 1.437, 0.658),
    '3': (0.777, 0.217, 0.744, 0.199),
    '4': (0.991, 0.2, 0.653, 0.136),
    '5': (0.817, 0.207, 0.935, 0.272),
    '6': (0.918, 0.216, 0.807, 0.152),
    '7': (0.962, 0.232, 1.171, 0.324),
    '8': (1.06, 0.139, 0.882, 0.233),
    '9': (0.842, 0.117, 0.977, 0.187),
    '=': (1.593, 0.168, 0.889, 0.076),
    '>': (0.774, 0.239, 0.524, 0.066),
    'A_cap': (1.221, 0.162, 0.87, 0.077),
    'B_cap': (0.883, 0.235, 0.712, 0.177),
    'C_cap': (1.055, 0.238, 0.887, 0.173),
    'D_cap': (1.294, 0.341, 0.791, 0.192),
    'Delta_cap': (0.837, 0.167, 0.803, 0.099),
    'E_cap': (1.074, 0.164, 0.898, 0.097),
    'F_cap': (1.438, 0.344, 1.077, 0.138),
    'G_cap': (0.929, 0.2, 1.163, 0.204),
    'Gamma_cap': (1.079, 0.191, 0.808, 0.144),
    'H_cap': (1.245, 0.125, 0.928, 0.055),
    'I_cap': (1.074, 0.215, 1.48, 0.254),
    'J_cap': (0.918, 0.244, 1.578, 0.186),
    'K_cap': (1.238, 0.334, 0.803, 0.173),
    'L_cap': (1.174, 0.117, 1.048, 0.251),
    'M_cap': (1.154, 0.124, 0.758, 0.059),
    'N_cap': (1.122, 0.274, 0.701, 0.174),
    'O_cap': (1.304, 0.296, 0.799, 0.142),
    'Omega_up': (0.87, 0.146, 1.47, 0.406),
    'P_cap': (0.991, 0.13, 0.631, 0.101),
    'Phi_up': (1.725, 0.303, 1.138, 0.249),
    'Pi_up': (1.132, 0.116, 1.041, 0.219),
    'Psi_up': (1.536, 0.196, 0.613, 0.242),
    'Q_cap': (0.751, 0.163, 0.934, 0.263),
    'R_cap': (1.095, 0.256, 1.045, 0.204),
    'S_cap': (1.077, 0.139, 0.817, 0.131),
    'Sigma_up': (1.004, 0.188, 1.297, 0.402),
    'T_cap': (0.915, 0.163, 1.44, 0.231),
    'U_cap': (1.237, 0.163, 1.309, 0.396),
    'V_cap': (1.051, 0.479, 1.266, 0.523),
    'W_cap': (0.966, 0.107, 1.048, 0.161),
    'Y_cap': (1.694, 0.553, 0.963, 0.45),
    'Z_cap': (1.006, 0.254, 0.887, 0.155),
    'a': (0.699, 0.136, 1.152, 0.198),
    'alpha': (0.833, 0.193, 1.213, 0.151),
    'b': (0.818, 0.15, 0.783, 0.156),
    'beta': (0.847, 0.243, 1.196, 0.226),
    'c': (0.992, 0.28, 1.214, 0.37),
    'cdot': (1.592, 0.518, 1.605, 0.523),
    'cup': (1.064, 0.055, 0.759, 0.05),
    'd': (0.991, 0.246, 1.167, 0.353),
    'delta': (0.875, 0.196, 0.959, 0.213),
    'div': (1.148, 0.202, 0.883, 0.199),
    'e': (1.09, 0.286, 0.905, 0.175),
    'epsilon': (0.978, 0.432, 1.135, 0.365),
    'f': (0.891, 0.169, 0.886, 0.166),
    'g': (1.371, 0.432, 1.109, 0.316),
    'gamma': (0.799, 0.313, 1.197, 0.316),
    'h': (0.868, 0.191, 0.986, 0.17),
    'i': (1.022, 0.171, 1.258, 0.27),
    'infty': (0.64, 0.18, 0.852, 0.217),
    'int': (0.929, 0.332, 0.69, 0.197),
    'j': (0.91, 0.193, 0.934, 0.27),
    'k': (0.789, 0.272, 1.046, 0.35),
    'l': (0.904, 0.17, 1.999, 0.346),
    'lambda': (0.716, 0.13, 0.683, 0.067),
    'm': (1.006, 0.208, 0.963, 0.21),
    'mu': (0.862, 0.279, 1.315, 0.279),
    'n': (1.076, 0.249, 1.101, 0.212),
    'nabla': (0.881, 0.315, 0.945, 0.235),
    'neq': (0.765, 0.082, 0.826, 0.247),
    'o': (1.115, 0.172, 0.786, 0.144),
    'omega': (0.886, 0.241, 0.87, 0.206),
    'p': (1.003, 0.228, 0.867, 0.129),
    'partial': (0.867, 0.235, 1.068, 0.315),
    'phi': (1.097, 0.332, 0.766, 0.185),
    'pi': (1.296, 0.462, 1.007, 0.228),
    'pm': (1.167, 0.174, 0.885, 0.116),
    'prime': (1.328, 0.169, 1.35, 0.109),
    'prod': (0.883, 0.151, 0.829, 0.18),
    'psi': (0.901, 0.176, 0.722, 0.273),
    'q': (1.22, 0.356, 1.102, 0.269),
    'r': (1.21, 0.249, 1.215, 0.298),
    'rbrace': (0.887, 0.206, 2.241, 0.227),
    's': (1.014, 0.161, 0.724, 0.161),
    'sigma': (0.906, 0.333, 1.171, 0.318),
    'sqrt': (0.826, 0.247, 0.951, 0.146),
    'sum': (0.722, 0.091, 0.858, 0.123),
    't': (0.901, 0.203, 1.424, 0.248),
    'theta': (0.772, 0.189, 1.076, 0.206),
    'times': (0.814, 0.17, 0.568, 0.122),
    'u': (0.944, 0.199, 1.258, 0.245),
    'v': (0.992, 0.234, 1.103, 0.289),
    'w': (1.073, 0.214, 0.937, 0.166),
    'x': (1.258, 0.28, 0.852, 0.222),
    'y': (1.161, 0.33, 0.896, 0.277),
    'z': (1.118, 0.307, 1.221, 0.327),
    '|': (0.784, 0.113, 3.242, 1.005),
}

# Symbols whose size is determined by content, not per-symbol stats
_SKIP_SIZE_CORRECTION = {"frac_bar", "sqrt"}


# ── Main augmentation ────────────────────────────────────────────────


def augment_bboxes(
    symbols: list[dict],
    tree: list[dict],
    sup_dy: float = -0.910,
    sup_dy_std: float = 0.300,
    sub_dy: float = 0.665,
    sub_dy_std: float = 0.250,
) -> list[dict] | None:
    """Perturb font bboxes toward handwritten proportions.

    Strategy:
    1. For each symbol, replace aspect ratio with handwritten one
       (keep center, adjust w/h).
    2. Add small position jitter.
    3. Tighten sibling y-alignment (handwritten is much tighter).
    4. Normalize to [0, 1].
    """
    n = len(symbols)
    if n == 0:
        return None

    # Copy font bboxes as [cx, cy, w, h]
    bboxes = []
    for s in symbols:
        x, y, w, h = s["bbox"]
        bboxes.append([x + w / 2, y + h / 2, w, h])

    # ── Step 1: Apply per-symbol size corrections ──────────────
    # Use a shared size factor (correlated h/w) + small aspect jitter
    for i in range(n):
        name = symbols[i]["name"]
        if name in _SKIP_SIZE_CORRECTION:
            continue
        stats = _HW_SYMBOL_STATS.get(name)
        if stats is None:
            continue
        h_med, h_std, w_med, w_std = stats
        # Shared size jitter (scales both dimensions together)
        rel_std = min(h_std / max(h_med, 1e-6), w_std / max(w_med, 1e-6))
        size_jitter = max(0.7, min(1.4, random.gauss(1.0, rel_std * 0.6)))
        # Small independent aspect ratio jitter
        aspect_jitter = max(0.85, min(1.15, random.gauss(1.0, 0.06)))
        bboxes[i][3] *= h_med * size_jitter
        bboxes[i][2] *= w_med * size_jitter * aspect_jitter

    # ── Step 1a2: Boost frac bar height to match HW ─────────────
    # HW frac bars are ~0.32× the height of their NUM/DEN children.
    # In font rendering bars are ultra-thin; boost to match HW proportions.
    for i in range(n):
        name = symbols[i]["name"]
        if name != "frac_bar":
            continue
        # Use NUM/DEN children heights as reference
        children = [j for j in range(n)
                    if tree[j]["parent"] == i and tree[j]["edge_type"] in (NUM, DEN)]
        if children:
            ref_h = sum(bboxes[j][3] for j in children) / len(children)
        else:
            continue
        target_h = ref_h * max(0.15, random.gauss(0.42, 0.12))
        if target_h > bboxes[i][3]:
            bboxes[i][3] = target_h
        # Clamp aspect ratio: HW min ~3.7, median ~11
        bar_w = bboxes[i][2]
        if bar_w > 0 and bar_w / bboxes[i][3] < 3.5:
            bboxes[i][3] = bar_w / 3.5

    # ── Step 1b: Clamp sup/sub size relative to parent ──────────
    # HW SUP: 90th pct=0.937, max=1.769 → keep 1.8
    # HW SUB: 90th pct=1.131, max=1.769 → clamp tighter to 1.2
    _MAX_H_RATIO = {SUP: 1.8, SUB: 1.2}
    for i in range(n):
        et = tree[i]["edge_type"]
        max_ratio = _MAX_H_RATIO.get(et)
        if max_ratio is None:
            continue
        p = tree[i]["parent"]
        if p < 0:
            continue
        parent_h = bboxes[p][3]
        child_h = bboxes[i][3]
        if parent_h > 0 and child_h / parent_h > max_ratio:
            s = (max_ratio * parent_h) / child_h
            bboxes[i][2] *= s
            bboxes[i][3] *= s

    # ── Step 1c: Resize sqrt to encompass content after swap ────
    children_by_et: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i in range(n):
        p = tree[i]["parent"]
        if p >= 0:
            children_by_et[p][tree[i]["edge_type"]].append(i)

    def _extent_right(idx):
        r = bboxes[idx][0] + bboxes[idx][2] / 2
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                r = max(r, _extent_right(ci))
        return r

    def _extent_left(idx):
        l = bboxes[idx][0] - bboxes[idx][2] / 2
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                l = min(l, _extent_left(ci))
        return l

    def _extent_top(idx):
        t = bboxes[idx][1] - bboxes[idx][3] / 2
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                t = min(t, _extent_top(ci))
        return t

    def _extent_bot(idx):
        b = bboxes[idx][1] + bboxes[idx][3] / 2
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                b = max(b, _extent_bot(ci))
        return b

    # ── Step 1c: Initial sqrt width clip (before gap squeezing) ────
    # Clip once now so step 2 gap squeezing sees a realistic sqrt width.
    # Re-clipped in step 2b after content positions are finalized.
    for p_i in range(n):
        content = children_by_et[p_i].get(SQRT_CONTENT, [])
        if not content:
            continue
        content_left = min(_extent_left(ci) for ci in content)
        content_right = max(_extent_right(ci) for ci in content)
        content_w = content_right - content_left
        sqrt_h = bboxes[p_i][3]
        hook = max(sqrt_h * 0.30, content_w * 0.25) * random.uniform(0.8, 1.2)
        sqrt_left = content_left - hook
        sqrt_right = content_right + sqrt_h * 0.03
        bboxes[p_i][0] = (sqrt_left + sqrt_right) / 2
        bboxes[p_i][2] = sqrt_right - sqrt_left

    # ── Step 1d: Squeeze frac num/den and bigop limits toward parent ──
    def _subtree_bot(idx):
        bot = bboxes[idx][1] + bboxes[idx][3] / 2
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                bot = max(bot, _subtree_bot(ci))
        return bot

    def _subtree_top(idx):
        top = bboxes[idx][1] - bboxes[idx][3] / 2
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                top = min(top, _subtree_top(ci))
        return top

    def _shift_subtree_y(idx, dy):
        bboxes[idx][1] += dy
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                _shift_subtree_y(ci, dy)

    def _shift_subtree_x(idx, dx):
        bboxes[idx][0] += dx
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                _shift_subtree_x(ci, dx)

    # ── Step 1d2: Scale down oversized subtrees inside SUP/SUB ───
    # A fraction (or other nested structure) inside a superscript should
    # have its entire subtree scaled so the visual extent matches HW.
    def _get_all_descendants(idx):
        result = []
        for ci in range(n):
            if tree[ci]["parent"] == idx:
                result.append(ci)
                result.extend(_get_all_descendants(ci))
        return result

    for i in range(n):
        et = tree[i]["edge_type"]
        if et not in (SUP, SUB):
            continue
        p = tree[i]["parent"]
        if p < 0:
            continue
        desc = _get_all_descendants(i)
        if not desc:
            continue
        all_nodes = [i] + desc
        sub_top = min(bboxes[j][1] - bboxes[j][3] / 2 for j in all_nodes)
        sub_bot = max(bboxes[j][1] + bboxes[j][3] / 2 for j in all_nodes)
        sub_h = sub_bot - sub_top
        # Use visual extent for frac_bar parents
        if symbols[p]["name"] == "frac_bar":
            p_h = _subtree_bot(p) - _subtree_top(p)
        else:
            p_h = bboxes[p][3]
        if p_h < 1e-6 or sub_h < 1e-6:
            continue
        # SUB should be tighter (HW SUB 90th pct = 1.131)
        if et == SUB:
            target_ratio = max(0.7, min(1.2, random.gauss(0.95, 0.20)))
        else:
            target_ratio = max(0.9, min(1.8, random.gauss(1.25, 0.25)))
        if sub_h / p_h > target_ratio:
            scale = (target_ratio * p_h) / sub_h
            sub_cx = sum(bboxes[j][0] for j in all_nodes) / len(all_nodes)
            sub_cy = sum(bboxes[j][1] for j in all_nodes) / len(all_nodes)
            for j in all_nodes:
                bboxes[j][0] = sub_cx + (bboxes[j][0] - sub_cx) * scale
                bboxes[j][1] = sub_cy + (bboxes[j][1] - sub_cy) * scale
                bboxes[j][2] *= scale
                bboxes[j][3] *= scale

    for p_i in range(n):
        p_top = bboxes[p_i][1] - bboxes[p_i][3] / 2
        p_bot = bboxes[p_i][1] + bboxes[p_i][3] / 2

        # Squeeze numerator toward bar
        num_kids = children_by_et[p_i].get(NUM, [])
        if num_kids:
            num_bot = max(_subtree_bot(i) for i in num_kids)
            avg_h = sum(bboxes[i][3] for i in num_kids) / len(num_kids)
            target = avg_h * 0.05
            gap = p_top - num_bot
            if gap > target:
                dy = (gap - target) * 0.85
                for i in num_kids:
                    _shift_subtree_y(i, dy)

        # Squeeze denominator toward bar
        den_kids = children_by_et[p_i].get(DEN, [])
        if den_kids:
            den_top = min(_subtree_top(i) for i in den_kids)
            avg_h = sum(bboxes[i][3] for i in den_kids) / len(den_kids)
            target = avg_h * 0.05
            gap = den_top - p_bot
            if gap > target:
                dy = (gap - target) * 0.85
                for i in den_kids:
                    _shift_subtree_y(i, -dy)

        # ── Fraction bar thickness variance ──
        # HW bar_hr std=0.142, AUG std=0.057 — add jitter
        if num_kids and den_kids:
            bar_h_jitter = random.gauss(1.0, 0.12)
            bboxes[p_i][3] *= max(0.7, min(1.3, bar_h_jitter))
            # Small left-right jitter on the fraction bar
            bar_dx = random.gauss(0, 0.03) * bboxes[p_i][2]
            bboxes[p_i][0] += bar_dx

        # Adjust sup/sub positions toward handwritten stats
        _SUP_SUB_TARGETS = {
            SUP: (sup_dy, sup_dy_std, 0.16, 0.15),  # (dy_med, dy_std, dx_in_ph, dx_std)
            SUB: (sub_dy, sub_dy_std, 0.12, 0.10),
        }
        # For frac_bar parents, use visual extent height instead of bar thickness
        if symbols[p_i]["name"] == "frac_bar" and (num_kids or den_kids):
            p_h = _subtree_bot(p_i) - _subtree_top(p_i)
        else:
            p_h = bboxes[p_i][3]
        p_cy = bboxes[p_i][1]
        p_right = bboxes[p_i][0] + bboxes[p_i][2] / 2  # cx + w/2 = right edge
        for et, (dy_med, dy_std, dx_med, dx_std) in _SUP_SUB_TARGETS.items():
            kids = children_by_et[p_i].get(et, [])
            if not kids or p_h < 1e-6:
                continue
            leftmost = min(kids, key=lambda i: bboxes[i][0])
            # Vertical: target center-to-center dy in parent_h units
            target_dy = random.gauss(dy_med, dy_std)
            target_dy = max(dy_med - 2 * dy_std, min(dy_med + 2 * dy_std, target_dy))
            cur_dy = (bboxes[leftmost][1] - p_cy) / p_h
            shift_y = (target_dy - cur_dy) * p_h
            # Horizontal: target gap in parent_h units
            target_dx = random.gauss(dx_med, dx_std)
            child_left = bboxes[leftmost][0] - bboxes[leftmost][2] / 2
            cur_dx = (child_left - p_right) / p_h
            shift_x = (target_dx - cur_dx) * 0.8 * p_h
            for i in kids:
                _shift_subtree_y(i, shift_y)
                _shift_subtree_x(i, shift_x)

        # Squeeze upper limits toward operator
        upper_kids = children_by_et[p_i].get(UPPER, [])
        if upper_kids:
            upper_bot = max(_subtree_bot(i) for i in upper_kids)
            avg_h = sum(bboxes[i][3] for i in upper_kids) / len(upper_kids)
            target = avg_h * 0.15
            gap = p_top - upper_bot
            if gap > target:
                dy = (gap - target) * 0.6
                for i in upper_kids:
                    _shift_subtree_y(i, dy)

        # Squeeze lower limits toward operator
        lower_kids = children_by_et[p_i].get(LOWER, [])
        if lower_kids:
            lower_top = min(_subtree_top(i) for i in lower_kids)
            avg_h = sum(bboxes[i][3] for i in lower_kids) / len(lower_kids)
            target = avg_h * 0.15
            gap = lower_top - p_bot
            if gap > target:
                dy = (gap - target) * 0.6
                for i in lower_kids:
                    _shift_subtree_y(i, -dy)

    # ── Step 2: Tighten sibling y-alignment + squeeze gaps ─────
    sibling_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, node in enumerate(tree):
        key = (node["parent"], node["edge_type"])
        sibling_groups[key].append(i)

    # Detect big operators (have UPPER or LOWER children)
    is_bigop = set()
    for i in range(n):
        p = tree[i]["parent"]
        if p >= 0 and tree[i]["edge_type"] in (UPPER, LOWER):
            is_bigop.add(p)

    # Build subtree extent: rightmost x of symbol and all its descendants
    def _subtree_right(idx):
        # Big ops: only use the symbol itself, not upper/lower bounds
        if idx in is_bigop:
            return bboxes[idx][0] + bboxes[idx][2] / 2
        right = bboxes[idx][0] + bboxes[idx][2] / 2
        for i in range(n):
            if tree[i]["parent"] == idx:
                right = max(right, _subtree_right(i))
        return right

    def _subtree_left(idx):
        if idx in is_bigop:
            return bboxes[idx][0] - bboxes[idx][2] / 2
        left = bboxes[idx][0] - bboxes[idx][2] / 2
        for i in range(n):
            if tree[i]["parent"] == idx:
                left = min(left, _subtree_left(i))
        return left

    for key, indices in sibling_groups.items():
        if len(indices) < 2:
            continue
        # Compute mean cy of group
        mean_cy = sum(bboxes[i][1] for i in indices) / len(indices)
        # Pull each sibling's cy toward the mean (tighten alignment)
        # HW sibling cy_std ≈ 0.096 — keep 95% of deviation
        for i in indices:
            old_cy = bboxes[i][1]
            new_cy = mean_cy + (old_cy - mean_cy) * 0.95
            dy = new_cy - old_cy
            _shift_subtree_y(i, dy)

        # Squeeze horizontal gaps between siblings (using subtree extents)
        sorted_idx = sorted(indices, key=lambda i: bboxes[i][0])

        # Record original group center before squeezing
        orig_center = sum(bboxes[i][0] for i in sorted_idx) / len(sorted_idx)

        _, parent_et = key
        for j in range(1, len(sorted_idx)):
            prev = sorted_idx[j - 1]
            curr = sorted_idx[j]
            prev_right = _subtree_right(prev)
            curr_left = _subtree_left(curr)
            gap = curr_left - prev_right
            # Use direct symbol height for NUM/DEN (subtree extent inflates it);
            # use subtree visual height for ROOT and other contexts
            if parent_et in (NUM, DEN):
                avg_h = (bboxes[prev][3] + bboxes[curr][3]) / 2
                target_gap = avg_h * max(0.0, random.gauss(0.40, 0.12))
            else:
                prev_vis_h = _subtree_bot(prev) - _subtree_top(prev)
                curr_vis_h = _subtree_bot(curr) - _subtree_top(curr)
                avg_h = (prev_vis_h + curr_vis_h) / 2
                target_gap = avg_h * max(0.0, random.gauss(0.25, 0.08))

            def _shift_subtree(idx, dx):
                bboxes[idx][0] += dx
                for ci in range(n):
                    if tree[ci]["parent"] == idx:
                        _shift_subtree(ci, dx)

            if gap > target_gap:
                # Too far: squeeze left
                shift = gap - target_gap
                for k in range(j, len(sorted_idx)):
                    _shift_subtree(sorted_idx[k], -shift)
            elif gap < target_gap:
                # Too tight: push right
                shift = target_gap - gap
                for k in range(j, len(sorted_idx)):
                    _shift_subtree(sorted_idx[k], shift)

        # Recenter group to preserve original center (fixes left-shift bias)
        # Add small random offset to match HW tendency (NUM≈-0.09, DEN≈-0.02)
        new_center = sum(bboxes[i][0] for i in sorted_idx) / len(sorted_idx)
        parent_idx, et = key
        if parent_idx >= 0 and et in (NUM, DEN):
            bar_w = bboxes[parent_idx][2]
            recenter_dx = orig_center - new_center + random.gauss(-0.05, 0.08) * bar_w
        else:
            recenter_dx = orig_center - new_center
        if abs(recenter_dx) > 1e-6:
            for i in sorted_idx:
                _shift_subtree(i, recenter_dx)

    # ── Step 2b: Re-clip sqrt width + close gaps with neighbours ──
    for p_i in range(n):
        content = children_by_et[p_i].get(SQRT_CONTENT, [])
        if not content:
            continue
        old_left = bboxes[p_i][0] - bboxes[p_i][2] / 2
        old_right = bboxes[p_i][0] + bboxes[p_i][2] / 2

        content_left = min(_subtree_left(ci) for ci in content)
        content_right = max(_subtree_right(ci) for ci in content)
        content_top = min(_subtree_top(ci) for ci in content)
        content_bot = max(_subtree_bot(ci) for ci in content)
        content_w = content_right - content_left
        content_h = content_bot - content_top
        sqrt_h = bboxes[p_i][3]

        # Width: hook on left, tiny margin on right
        hook = max(sqrt_h * 0.30, content_w * 0.25) * random.uniform(0.8, 1.2)
        sqrt_left = content_left - hook
        sqrt_right = content_right + sqrt_h * 0.03
        bboxes[p_i][0] = (sqrt_left + sqrt_right) / 2
        bboxes[p_i][2] = sqrt_right - sqrt_left

        # Height: clip to content extent + overline margin above
        overline = content_h * random.uniform(0.10, 0.25)
        sqrt_top = content_top - overline
        sqrt_bot = content_bot + content_h * 0.03
        bboxes[p_i][1] = (sqrt_top + sqrt_bot) / 2
        bboxes[p_i][3] = sqrt_bot - sqrt_top

        # Close right-side gap: if sqrt shrank, pull subsequent siblings left
        right_delta = sqrt_right - old_right  # negative = sqrt shrank on right
        if right_delta < 0:
            p_parent = tree[p_i]["parent"]
            p_et = tree[p_i]["edge_type"]
            siblings = sibling_groups.get((p_parent, p_et), [])
            sorted_sibs = sorted(siblings, key=lambda i: bboxes[i][0])
            my_pos = next((k for k, s in enumerate(sorted_sibs) if s == p_i), -1)
            if my_pos >= 0:
                for k in range(my_pos + 1, len(sorted_sibs)):
                    _shift_subtree(sorted_sibs[k], right_delta)

    # ── Step 2c: Final SUP/SUB height clamp ─────────────────────
    # Later steps (sqrt resize, etc.) can change parent heights, so re-clamp.
    # Run twice to handle nested SUP/SUB (parent shrink → child ratio increases).
    _FINAL_MAX_RATIO = {SUP: 1.8, SUB: 1.2}
    for _pass in range(2):
        for i in range(n):
            et = tree[i]["edge_type"]
            max_r = _FINAL_MAX_RATIO.get(et)
            if max_r is None:
                continue
            p = tree[i]["parent"]
            if p < 0:
                continue
            parent_h = bboxes[p][3]
            child_h = bboxes[i][3]
            if parent_h > 0 and child_h / parent_h > max_r:
                s = (max_r * parent_h) / child_h
                bboxes[i][2] *= s
                bboxes[i][3] *= s

    # ── Step 3: Convert back to [x, y, w, h] and normalize ──────
    for i in range(n):
        cx, cy, w, h = bboxes[i]
        bboxes[i] = [cx - w / 2, cy - h / 2, w, h]

    all_x = [b[0] for b in bboxes]
    all_y = [b[1] for b in bboxes]
    all_x2 = [b[0] + b[2] for b in bboxes]
    all_y2 = [b[1] + b[3] for b in bboxes]

    xmin = min(all_x)
    ymin = min(all_y)
    xmax = max(all_x2)
    ymax = max(all_y2)
    ref = max(xmax - xmin, ymax - ymin, 1e-6)

    result = []
    for i in range(n):
        bx, by, bw, bh = bboxes[i]
        result.append({
            "name": symbols[i]["name"],
            "bbox": [
                round((bx - xmin) / ref, 6),
                round((by - ymin) / ref, 6),
                round(bw / ref, 6),
                round(bh / ref, 6),
            ],
        })

    return result
