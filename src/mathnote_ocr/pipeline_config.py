"""Pipeline configuration loader.

Usage:
    cfg = load_config("default")                       # bundled config
    cfg = load_config("configs/my_experiment.yaml")    # explicit path
"""

from pathlib import Path
import yaml

_PACKAGED_CONFIGS_DIR = Path(__file__).parent / "configs"


def _resolve_config_path(name_or_path) -> Path:
    """Bundled name (no separator, no .yaml suffix) → packaged dir.
    Otherwise treat as a filesystem path (absolute or CWD-relative).
    """
    s = str(name_or_path)
    if "/" in s or "\\" in s or s.endswith((".yaml", ".yml")):
        return Path(s)
    return _PACKAGED_CONFIGS_DIR / f"{s}.yaml"


def load_config(name_or_path):
    """Load a config by bundled name or explicit path. None → empty dict."""
    if name_or_path is None:
        return {}
    path = _resolve_config_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get(cfg, dotpath, default=None):
    """Get a value by dotpath (e.g. 'classifier.run'). Returns default if missing."""
    node = cfg
    for key in dotpath.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default
