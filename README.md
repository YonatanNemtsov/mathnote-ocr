# MathNote OCR

Stroke-based handwritten math to LaTeX OCR.

Unlike image-based OCR, this pipeline takes **pen stroke data** (sequences of `(x, y)` points from a tablet or touchscreen) and produces LaTeX. Stroke input is richer than pixels — it preserves timing, order, and direction, which helps disambiguate visually similar symbols.

## Architecture

```
Strokes → Grouper → Classifier → Tree Parser → LaTeX
         (group strokes (classify each  (predict parent+
         into symbols)   symbol)         edge for each symbol)
```

- **Grouper** — partitions strokes into symbols (classical, neighbor-based)
- **Classifier** — 128×128 CNN with prototype-based OOD detection
- **Tree Parser** — 8-symbol transformer that predicts partial parent/edge relationships for spatial subsets; predictions are aggregated into evidence, then a beam-search builder reconstructs the tree
- **GNN Verifier** — refines tree candidates during beam search

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