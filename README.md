# MathNote OCR

A handwriting (stroke data) to latex recognition system, designed for real-time interactive use. It supports incremental prediction, where strokes can be added or removed, and previous classification work is reused. It also supports interactive correction: a subexpression (subtree) can be "pinned", so the subsequent prediction keeps that subtree structure and builds on top of it.

Total model size in the current default configuration is ~1.2M parameters: a 0.65M symbol classifier (CNN), a 0.43M transformer that runs on subsets of input strokes, and a 0.13M GNN that refines the evidence from the transformer. Runs on CPU.


https://github.com/user-attachments/assets/9d3e37ca-8e27-4d48-9351-22ee9f3a0744



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

#### Pinning a subtree

Given a set of strokes for detection, a subset of those strokes can be "pinned" to a specific subexpression (subtree), so that subsequent detections keep that subtree structure fixed. This is usefull for interactive correction, for example, locking in a correctly recognized fraction while continuing to draw the rest of the expression.


```python
from mathnote_ocr import PinnedTree, PinSymbol, PinEdge, Edge

session = ocr.session()
session.add_stroke([(10, 20), (15, 25)])      # the "x"
session.add_stroke([(30, 10), (35, 15)])      # the "2"

pin = PinnedTree.build(
    symbols=[
        PinSymbol("x", [session.strokes[0]]),
        PinSymbol("2", [session.strokes[1]]),
    ],
    edges=[PinEdge(parent=0, child=1, edge=Edge.SUP)],
)

expr = session.detect(pins=[pin])   # the x^2 subtree is preserved
```

Pins are per-call inputs to `detect()`; the caller controls their lifecycle. Indices in `edges` refer to positions in the `symbols` list. Pins may leave the symbol set as a forest (no edges, or multiple local roots); in that case a synthetic `expr` node is inserted as their common parent.

#### Matrices and cases

Mark a region's strokes as a grid and it is split into cells and read as one
item of the surrounding expression:

```python
from mathnote_ocr import Structure

expr = ocr.detect(strokes, structures=[Structure("grid", tuple(matrix_stroke_ids))])
for atom_id, grid in expr.grids.items():
    print(grid.env, grid.shape, grid.cell_latex)   # pmatrix (2, 2) (('1', '2'), ('3', '4'))
```

Rows are lines of writing and columns line up across rows, so a matrix still
being written reads with empty cells. Each cell is read like an expression on
its own (`grid.cell_trees`); the brackets stay symbols of the expression around
the grid, so `(…)^2` reads as it would anywhere.

### Configuring the engine for an app

An app usually knows more than the engine: which symbols it can use, which
relations never occur, what a well-formed reading is. The engine provides the
mechanisms; the app states the rules. All three are given once to `MathOCR`
(or per call to `detect`):

```python
from mathnote_ocr import Grammar, MathOCR, Relations, Rewrite, Vocabulary

ocr = MathOCR(
    config,
    vocabulary=Vocabulary(exclude={"alpha", "beta"}, aliases={"slash": ","}),
    relations=Relations(exclude={"sub"}),          # no subscripts in this app
    grammar=Grammar(no_two_operators, rewrites=[Rewrite(("-", "-"), ("=",))]),
)
```

- **`Vocabulary`**: symbols the app never expects are removed from the
  classifier's choices (their probability dropped, the rest not rescaled);
  `aliases` read one class as another.
- **`Relations`**: relations the tree may not have (`"sup"`, `"sub"`, `"num"`,
  `"den"`, …) get no votes: the parser reads the most probable tree among the
  allowed ones.
- **`Grammar`**: a rule is a function `(names, where[, boxes]) -> positions it
  objects to`. It sees one line of the reading at a time: the siblings under one
  parent and relation, in order, named as shown; `where` says which line it is
  (`"main"`, `"sup"`, `"num"`, `"den"`, `"cell"`, …), and a rule taking `boxes`
  also gets each symbol's box. A fraction's empty side is a line of its own, and
  position `-1` objects to its bar. A reading that breaks a rule is repaired
  locally: each flagged symbol is re-read on its own (as one of its other
  readings, or placed elsewhere, or by the app's own move) while everything
  else stays as read, and a repair is kept only if fewer rules are broken.
  A `Rewrite(parts, into, min_confidence=0)` joins two symbols that are really
  one (an `=` written as two strokes and split) when the classifier reads their
  strokes together as a target, and only if that fixes something.

### Web interface

The bundled demo:

```bash
python demos/web/server.py
```

Open [http://localhost:8080](http://localhost:8080). Draw math expressions; get LaTeX.

### Collect handwritten training data

```bash
python web_tools/collect_expr_server.py
```

Open `web_tools/collect_expr.html` in your browser (pure WebSocket server on port 8770). Draw expressions matching sampled LaTeX prompts; data saves to `data/shared/tree_handwritten/`.

## Installation

Requires Python 3.10+.

```bash
pip install mathnote-ocr
```

Default production weights (mixed_v10 subset + GNN, v9_combined classifier) are bundled with the package — no download needed.

Note: on Linux, pip pulls the CUDA build of PyTorch (~2.5GB). The model runs on CPU — for a lighter install, get CPU-only torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### From source

```bash
git clone https://github.com/YonatanNemtsov/mathnote-ocr.git
cd mathnote-ocr
pip install -e .           # library only
pip install -e .[tools]    # + tool servers (web UI, data collection)
uv sync --group dev        # development setup with linting
```



## Architecture

```
strokes → groups of strokes → classified symbols → expression tree → LaTeX
            Grouper               Classifier          Tree Parser
```

**Grouper** partitions the strokes into symbols. One symbol can span multiple strokes (e.g. "T" has two, "\Pi" might have 2 or 3 etc). The grouper enumerates plausible groupings based on spatial proximity and returns the top-K candidates.

**Classifier** is a 32×32 CNN that labels each candidate symbol. It also computes a prototype distance for every class, which flags nonsense groupings as out-of-distribution — this lets the grouper reject bad partitions.

**Tree Parser** builds a structured expression tree from the labeled symbols. Each node has a parent, an edge type (superscript, subscript, numerator, denominator, etc.), and a sibling order. Internally:

- A small fixed-size transformer (the *subset model*) is run on overlapping subsets of 3–8 symbols at a time and predicts a partial tree for each subset.
- These predictions are aggregated into an *evidence graph* — a dense tensor of parent/edge votes for every symbol pair.
- A graph neural network (the *EvidenceGNN*) refines that evidence using edge-feature-biased attention. Unlike the subset model, the GNN operates on the full expression regardless of size.
- A bottom-up beam-search builder consumes the refined scores, builds the tree leaf-by-leaf, branches on uncertain assignments, and picks the highest-scoring result.

The tree is then rendered to LaTeX.

## Configuration

The pipeline is configured via YAML files. `MathOCR()` uses the bundled `default.yaml`.

Note: it is advised to use the default config if you are not experimenting or trying to modify and extend the project. Most of the non-default config options are experimental, and are a result of trying different tree building algorithms and hyperparameters. They are not end user configs.

To use a different config:

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
