from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
SYMBOLS_DIR = DATA_DIR / "shared" / "symbols"
AUGMENTED_DIR = DATA_DIR / "shared" / "classifier_augmented"
WEIGHTS_DIR = ROOT_DIR / "weights"

# ── Seed ─────────────────────────────────────────────────────────────

SEED = 42

# ── Rendering ────────────────────────────────────────────────────────

RENDER_CANVAS_SIZE = 128
RENDER_STROKE_WIDTH = 2.0
RENDER_PADDING_RATIO = 0.15
