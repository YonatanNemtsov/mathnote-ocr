"""Pipeline configuration loader.

Load named YAML configs from configs/ directory.
Priority: CLI arg > config file > hardcoded default.

Usage:
    cfg = load_config("default")        # loads configs/default.yaml
    run = get(cfg, "classifier.run", "v4")  # dotpath access with fallback
"""

from pathlib import Path
import yaml

_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(name):
    """Load a named config. Returns dict, or empty dict if name is None."""
    if name is None:
        return {}
    path = _CONFIGS_DIR / f"{name}.yaml"
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
