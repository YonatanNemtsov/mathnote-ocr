# MathNote OCR

Stroke-based handwritten math to LaTeX OCR.

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

## Installation

Requires Python 3.10.

```bash
git clone https://github.com/YonatanNemtsov/mathnote-ocr.git
cd mathnote-ocr
pip install -r requirements.txt
```

The repo includes default production weights (mixed_v10 subset + GNN, v9_combined classifier). No download needed.

## Usage

### Run the web interface

```bash
python3.10 tools/web/server.py --config mixed_v10_backtrack_gnn
```

Open [http://localhost:8768](http://localhost:8768) in your browser. Draw math expressions; get LaTeX.

### Collect your own handwritten training data

```bash
python3.10 tools/collect_expr_server.py
```

Then open `tools/collect_expr.html` in your browser (the server is a pure WebSocket server on port 8770; the HTML file connects to it). Draw expressions matching sampled LaTeX prompts; data saves to `data/shared/tree_handwritten/`.

### Python API

```python
from api import MathOCR

ocr = MathOCR(config="mixed_v10_backtrack_gnn")

# strokes: list of strokes, each stroke is a list of {"x": float, "y": float} points
strokes = [
    [{"x": 10, "y": 20}, {"x": 15, "y": 25}, ...],  # first stroke
    [{"x": 30, "y": 20}, {"x": 35, "y": 25}, ...],  # second stroke
    ...
]

results = ocr.parse(strokes, canvas_size=800)
# Returns [{"latex": "...", "confidence": 0.89, "symbols": [...]}]

print(results[0]["latex"])  # → "x^{2} + y"
```

Between unrelated inputs, call `ocr.clear()` to reset the grouper cache. The cache is designed for incremental (interactive) use where strokes are added one by one.

## Training from scratch

```bash
# 1. Generate synthetic training data (~100MB, a few minutes)
python3.10 data/runs/tree_subset/mixed_v10/build.py

# 2. Train the subset tree parser
python3.10 tree_parser/subset_train.py --run my_v1

# 3. Generate GNN evidence data (needs trained subset model)
python3.10 data/runs/gnn/mixed_v10/build_mixed_v10.py

# 4. Train the GNN
python3.10 tree_parser/gnn/train.py --run my_v1
```

The classifier trains on included handwritten symbol JSONs in `data/shared/symbols/`:

```bash
python3.10 classifier/train.py --run my_classifier
```

## Repository structure

```
tree_parser/      # Tree parser: subset model + GNN + beam search builder
classifier/       # CNN symbol classifier with prdototype OOD
engine/           # Grouper, stroke rendering, checkpoints
grouper_gnn/      # GNN-based grouper (optional alternative to classical)
data_gen/         # Synthetic expression generators (v1–v16 + v3 tree-first)
latex_utils/      # LaTeX rendering, glyph definitions, expression utilities
configs/          # YAML pipeline configs
scripts/          # Evaluation and diagnostic scripts
tools/            # Web servers: collection, testing, inference UI
data/shared/      # Handwritten dataset (5,672 symbols, 300+ expressions)
weights/          # Default model checkpoints
```

## License

Apache 2.0