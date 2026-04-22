# MathNote OCR

Stroke-based handwritten math to LaTeX OCR.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/YonatanNemtsov/mathnote-ocr.git
cd mathnote-ocr
pip install -e .
```

For development setup with linting:

```bash
uv sync --group dev        # or: pip install -e .[dev]
```

To also get the tool servers (web UI, data collection):

```bash
pip install -e .[tools]
```

Default production weights (mixed_v10 subset + GNN, v9_combined classifier) are bundled with the package — no download needed.

## Usage

### Python API

```python
from mathnote_ocr import MathOCR

ocr = MathOCR()  # bundled default config + weights

strokes = [
    [(10, 20), (15, 25), ...],  # first stroke — list of (x, y) tuples
    [(30, 20), (35, 25), ...],  # second stroke
]

expr = ocr.detect(strokes)
print(expr.latex)         # → "x^{2} + y"
print(expr.confidence)    # → 0.89
for sym in expr:
    print(sym.name, sym.bbox)
```

`detect()` returns an `Expression` with:

- `.latex` — the recognized LaTeX string
- `.confidence` — overall confidence (0–1)
- `.symbols` — `dict[int, DetectedSymbol]` keyed by symbol id
- `.tree` — the parsed expression tree
- `.strokes` — the input strokes (with stable ids)
- `.alternatives` — top-k alternative Expressions (when `top_k > 1`)

#### Interactive / incremental detection

For a drawing UI that adds strokes one by one:

```python
session = ocr.session()

session.add_stroke([(10, 20), (15, 25)])      # auto-id
session.add_stroke([(30, 20), (35, 25)])

expr = session.detect()
print(expr.latex)

session.remove_stroke(0)                       # undo last stroke
expr = session.detect()
```

The session keeps an incremental cache so repeated `detect()` calls on overlapping stroke sets are fast.

### Web interface

The bundled demo:

```bash
python demos/web/server.py
```

Open [http://localhost:8080](http://localhost:8080). Draw math expressions; get LaTeX.

### Collect handwritten training data

```bash
python tools/collect_expr_server.py
```

Open `tools/collect_expr.html` in your browser (pure WebSocket server on port 8770). Draw expressions matching sampled LaTeX prompts; data saves to `data/shared/tree_handwritten/`.

## Architecture

```
strokes → groups of strokes → classified symbols → expression tree → LaTeX
            Grouper               Classifier          Tree Parser
```

**Grouper** partitions the strokes into symbols. One symbol can span multiple strokes (e.g. `=` has two). The grouper enumerates plausible groupings based on spatial proximity and returns the top-K candidates.

**Classifier** is a 128×128 CNN that labels each candidate symbol. It also computes a prototype distance for every class, which flags nonsense groupings as out-of-distribution — this lets the grouper reject bad partitions.

**Tree Parser** builds a structured expression tree from the labeled symbols. Each node has a parent, an edge type (superscript, subscript, numerator, denominator, etc.), and a sibling order. Internally:

- A small fixed-size transformer (the *subset model*) is run on overlapping subsets of 3–8 symbols at a time and predicts a partial tree for each subset.
- These predictions are aggregated into an *evidence graph* — a dense tensor of parent/edge votes for every symbol pair.
- A graph neural network (the *EvidenceGNN*) refines that evidence using edge-feature-biased attention. Unlike the subset model, the GNN operates on the full expression regardless of size.
- A bottom-up beam-search builder consumes the refined scores, builds the tree leaf-by-leaf, branches on uncertain assignments, and picks the highest-scoring result.

The tree is then rendered to LaTeX.

## Configuration

The pipeline is configured via YAML files. `MathOCR()` uses the bundled `default.yaml`. To use a different config:

```python
ocr = MathOCR(config="configs/mixed_v9_backtrack.yaml")
```

### Config structure

A pipeline config has three sections, one per stage:

```yaml
classifier:
  run: v9_combined          # which classifier checkpoint to load
  ood_threshold: 15.0       # reject grouping if prototype distance exceeds this
  min_confidence: 0.15
  per_class_thresholds:     # per-class OOD overrides (useful for ambiguous symbols)
    x: 15.0
    dot: 15.0
    "-": 15.0

grouper:
  top_k: 3                  # number of candidate partitions to return
  max_strokes_per_symbol: 4
  size_multiplier: 0.1      # neighbour distance relative to stroke diagonal
  min_merge_distance: 14.0
  max_group_diameter_ratio: 2.2
  conflict_threshold: 0.32

tree_parser:
  subset_run: mixed_v10     # subset model checkpoint
  gnn_run: mixed_v10        # GNN refinement checkpoint (optional — omit for subset-only)
  tree_strategy: backtrack  # "backtrack" (beam search) or "edmonds"
  scoring: full_spatial
  tta_runs: 1               # test-time augmentation — jitter bboxes and average
  tta_dx: 0.05
  tta_dy: 0.05
  tta_size: 0.05
  root_discount: 0.3
```

### Where weights are looked up

Any `run` field (e.g., `classifier.run`, `tree_parser.subset_run`, `tree_parser.gnn_run`) can be either:

- **A name** like `v9_combined`: looked up as `{weights_dir}/{model_type}/{run_name}/checkpoint.pth`, falling back to the bundled weights if not found there.
- **A path** like `./my_weights/classifier_final.pth` (ends in `.pth` or contains `/`): loaded directly.

So you can mix and match:

```python
# All bundled
ocr = MathOCR()

# Custom classifier, bundled tree parser (uses fallback)
ocr = MathOCR(
    classifier_run="my_v1",
    weights_dir="./my_weights",   # has classifier/my_v1/ but not tree_subset/
)

# Point runs directly at files, no weights_dir needed
ocr = MathOCR(
    classifier_run="./models/my_classifier.pth",
    subset_run="./models/my_subset.pth",
)
```

### Bundled vs repo configs

- **`src/mathnote_ocr/configs/default.yaml`** — ships with the package, used when you call `MathOCR()` or `MathOCR(config="default")`.
- **`configs/*.yaml`** — experimental/alternative configs tracked in the repo (mixed_v9 variants, bottomup, backtrack_collapse, etc.). Reference them by path: `MathOCR(config="configs/mixed_v9_backtrack.yaml")`.

### Full field reference

See [`configs/reference.yaml`](configs/reference.yaml) — every supported field with its default and a one-line description. Fields marked `[experimental]` can change or be removed without notice; the rest are stable.

### Creating a new config

Copy an existing one (start with `configs/reference.yaml`), adjust the fields you need, reference it by path. No schema validation — unrecognized fields are silently ignored, missing ones fall back to defaults.

## Training from scratch

```bash
# 1. Generate synthetic training data (~100MB, a few minutes)
python data/runs/tree_subset/mixed_v10/build.py

# 2. Train the subset tree parser
python -m mathnote_ocr.tree_parser.subset_train \
    --run my_v1 \
    --train data/runs/tree_subset/mixed_v10/train.jsonl \
    --val data/runs/tree_subset/mixed_v10/val.jsonl

# 3. Generate GNN evidence data (needs a trained subset model)
python data/runs/gnn/mixed_v10/build_mixed_v10.py

# 4. Train the GNN (looks up data under data/runs/gnn/{subset-run}/)
python -m mathnote_ocr.tree_parser.gnn.train \
    --run my_v1 \
    --subset-run mixed_v10 \
    --train-data train \
    --val-data val
```

The classifier trains on included handwritten symbol JSONs in `data/shared/symbols/`:

```bash
python -m mathnote_ocr.classifier.train --run my_classifier
```

## Repository structure

```
src/mathnote_ocr/     # The package
  api.py              # Public API (MathOCR class)
  tree_parser/        # Subset model + GNN + beam search builder
  classifier/         # CNN classifier with prototype OOD
  engine/             # Grouper, stroke rendering, checkpoints
  grouper_gnn/        # GNN-based grouper (optional alternative)
  data_gen/           # Synthetic expression generators
  latex_utils/        # LaTeX rendering, glyphs
  weights/            # Bundled default checkpoints
  configs/            # Bundled default YAML config

configs/              # Experiment configs (mixed_v9_*, mixed_v10_*)
weights/              # User-trained checkpoints (development)
data/                 # Training data
scripts/              # Evaluation and diagnostic scripts
tools/                # Web servers (inference UI, collection)
```

## License

Apache 2.0
