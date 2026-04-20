import random

from mathnote_ocr.data_gen.latex_sampling import v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16

# (module, weight)
#   v1  balanced recursive          v9  juxtaposition-heavy
#   v2  context-aware misc symbols  v10 kitchen sink
#   v3  structure-dense / deep      v11 hard: structural ambiguities
#   v4  wide & flat / many siblings v12 missing patterns from v1-v11
#   v5  frac & index heavy          v13 long num/den, nested fracs
#   v6  maximum nesting             v14 hard: nested frac/int/abs
#   v7  academic / calculus          v15 complex num/den with integrals
#   v8  discrete math / relations   v16 superscript + juxtaposition
_VERSIONS = [
    (v1,  10),   # balanced baseline
    (v2,   6),   # context-aware misc
    (v3,   6),   # structure-dense
    (v4,   6),   # wide & flat
    (v5,   8),   # frac & index heavy
    (v6,   4),   # max nesting
    (v7,   6),   # calculus
    (v8,   4),   # discrete math
    (v9,   6),   # juxtaposition
    (v10,  6),   # kitchen sink
    (v11,  8),   # hard: ambiguities
    (v12,  6),   # missing patterns
    (v13,  6),   # long num/den
    (v14,  8),   # hard: nested frac/int/abs
    (v15,  6),   # complex num/den
    (v16,  8),   # sup + juxtaposition
]

_modules = [m for m, _ in _VERSIONS]
_weights = [w for _, w in _VERSIONS]


VERSION_NAMES = {
    1: "balanced baseline",
    2: "context-aware misc",
    3: "structure-dense",
    4: "wide & flat",
    5: "frac & index heavy",
    6: "max nesting",
    7: "calculus",
    8: "discrete math",
    9: "juxtaposition",
    10: "kitchen sink",
    11: "hard: ambiguities",
    12: "missing patterns",
    13: "long num/den",
    14: "hard: nested frac/int/abs",
    15: "complex num/den",
    16: "sup + juxtaposition",
}


from mathnote_ocr.data_gen.latex_sampling.symbols import clean_latex


def sample_all():
    mod = random.choices(_modules, weights=_weights)[0]
    return clean_latex(mod.sample())


def sample_all_with_gen3():
    """50/50 mix of dg_all (v1-v16) and gen3 (tree-first)."""
    if random.random() < 0.5:
        return sample_all()
    from mathnote_ocr.data_gen.latex_sampling_v3.generator import sample as gen3_sample
    return gen3_sample()


def sample_version(n: int):
    """Sample from a specific version (1-indexed)."""
    return _modules[n - 1].sample()
