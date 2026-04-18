"""Expression sampler registry — select and invoke template-based samplers."""


# ── Sampler registry ─────────────────────────────────────────────────

# (module_path:func_name, description)
_SAMPLERS: dict[str, tuple[str, str]] = {
    "dg_all": ("data_gen:sample_all", "all (weighted mix)"),
    "dg1":  ("data_gen.latex_sampling.v1:sample",  "balanced baseline"),
    "dg2":  ("data_gen.latex_sampling.v2:sample",  "context-aware misc"),
    "dg3":  ("data_gen.latex_sampling.v3:sample",  "structure-dense"),
    "dg4":  ("data_gen.latex_sampling.v4:sample",  "wide & flat"),
    "dg5":  ("data_gen.latex_sampling.v5:sample",  "frac & index heavy"),
    "dg6":  ("data_gen.latex_sampling.v6:sample",  "max nesting"),
    "dg7":  ("data_gen.latex_sampling.v7:sample",  "calculus"),
    "dg8":  ("data_gen.latex_sampling.v8:sample",  "discrete math"),
    "dg9":  ("data_gen.latex_sampling.v9:sample",  "juxtaposition"),
    "dg10": ("data_gen.latex_sampling.v10:sample", "kitchen sink"),
    "dg11": ("data_gen.latex_sampling.v11:sample", "hard: ambiguities"),
    "dg12": ("data_gen.latex_sampling.v12:sample", "missing patterns"),
    "dg13": ("data_gen.latex_sampling.v13:sample", "long num/den"),
    "dg14": ("data_gen.latex_sampling.v14:sample", "hard: nested frac/int/abs"),
    "dg15": ("data_gen.latex_sampling.v15:sample", "complex num/den"),
    "dg16": ("data_gen.latex_sampling.v16:sample", "sup + juxtaposition"),
    "gen":  ("data_gen.latex_sampling_v2.generator:sample", "generator v2"),
    "gen3": ("data_gen.latex_sampling_v3.generator:sample", "generator v3 (tree-first)"),
    "dg_all_gen3": ("data_gen:sample_all_with_gen3", "dg_all + gen3 (50/50)"),
}

# Module-level sampler function, set by _set_sampler()
_sampler_fn = None
_sampler_name: str | None = None


def _set_sampler(name: str) -> None:
    """Import and set the active sampler by registry name."""
    global _sampler_fn, _sampler_name
    path, func_name = _SAMPLERS[name][0].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(path)
    _sampler_fn = getattr(mod, func_name)
    _sampler_name = name


def sampler_list() -> dict[str, str]:
    """Return {name: description} for all registered samplers."""
    return {name: desc for name, (_, desc) in _SAMPLERS.items()}


def sample_expression() -> str:
    """Sample using the active sampler (default: dg_all)."""
    global _sampler_fn
    if _sampler_fn is None:
        _set_sampler("dg_all")
    from data_gen.latex_sampling.symbols import clean_latex
    return clean_latex(_sampler_fn())
