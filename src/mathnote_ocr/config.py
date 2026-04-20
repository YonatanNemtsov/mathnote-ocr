"""Package-level config: bundled weights path and rendering constants.

Repo-specific data paths (training data, collection output, etc.) are not
defined here — tools and scripts take them as CLI args with CWD-relative
defaults (typically data/shared/symbols, weights/, configs/).
"""

from pathlib import Path

# Bundled default weights that ship with the package
WEIGHTS_DIR = Path(__file__).parent / "weights"

# Seed
SEED = 42

# Rendering
RENDER_CANVAS_SIZE = 128
RENDER_STROKE_WIDTH = 2.0
RENDER_PADDING_RATIO = 0.15
