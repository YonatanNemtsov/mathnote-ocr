# Symbol vocabulary (2026-09-26, decided 2026-09-27)

> **Decided:** the study vocabulary is the current 125 + all of Tier 1 +
> `⋮` + `ℝ ℕ ℤ ℚ ℂ` (34 new classes) + all five accents (`\hat \bar \vec
> \dot \tilde`, labelled as marks `accent_*`) = 164 classes. It lives in
> the handwriting study app's `study/vocabulary.json`; the new classes are in
> `glyphs.SYMBOL_TO_LATEX` and `align.latex_to_labels` knows accents,
> `\mathbb` and `\to`. Glyph-vs-meaning (§2) and the accent structure (§4)
> are decided later, from data — the study labels by meaning, which
> supports either. Rest of this file: the original proposal.

What the engine should be able to read, decided *before* collecting
multi-writer data: the study's prompts, the classifier's classes and the
parser's rules all follow from it. Mark each line: keep `[x]`, drop `[ ]`.

## 1. Where we are

- **125 classes**, trained on **~7,800 samples, all one writer**: median 32
  isolated + 8 cut from expressions per class; capitals and several
  symbols only ~20–25 (C K Z B S D O Y Q, Δ Ω Ψ, ∈ ≥ ± ∂ ∀ `,` `}` `>`).
- That volume and single writer explain most of the failure on other
  people's handwriting. The study (multi-writer, prompted, aligned) is the
  main fix; this document decides *what* it collects.

## 2. Principle to decide: glyph vs. meaning

Some classes are the **same shape**, told apart only by size or position —
which the classifier never sees (it gets each symbol cropped and scaled):

| Same glyph | Told apart by |
|---|---|
| `c C`, `o O`, `s S`, `p P`, `u U`, `v V`, `w W`, `x X`, `z Z`, `k K` | size relative to neighbours |
| `o O 0` | size, context (digits around?) |
| `x X ×` | size, position between operands |
| `. ·` (dot, cdot) | height on the line |
| `1 l \|` (and `I`) | context, length |
| `- frac_bar` (+ bar accent, see 4) | already structural: the parser decides |
| `sum Σ`, `prod Π` | role (limits / operands) |
| `∈ ε` | context |

**Option A (today):** one class per meaning; the classifier guesses from
the crop alone → systematic confusions, worse for strangers (their size
habits differ from yours).
**Option B (proposed):** the classifier predicts the **glyph family**; a
context step decides the meaning from size/position/neighbours — the
same move as `frac_bar` vs `-` today, generalised. This is what the
shelved `label-refiner` branch started. Fewer, cleaner classes to learn;
the ambiguity goes where the information is.

- [ ] Adopt option B (glyph families + context resolution)

## 3. The current 125

Keep all unless marked; notes on weak or questionable ones.

- **Digits** `0–9` — [x]
- **Latin lower** `a–z` — [x]
- **Latin upper** `A–Z` — [x] (glyph families with lower case where the shape matches, option B)
- **Greek lower** α β γ δ ε θ λ μ π σ φ ψ ω — [x]
- **Greek upper** Γ Δ Π Σ Φ Ψ Ω — [x] (Σ/Π share glyphs with ∑/∏)
- **Operators** `+ - ± × ÷ ·` , `/` (slash) — [x]
- **Relations** `= ≠ < > ≤ ≥` — [x]
- **Sets & logic** `∈ ⊂ ∩ ∪ ∀ ∃` — [x]
- **Calculus** `∫ ∑ ∏ ∂ ∇ √ ∞` , prime `′` — [x]
- **Delimiters** `( ) [ ] { } |` — [x]
- **Punctuation** `, ; : ! .` , `…` (ldots) — [x]
- **Arrows** `→` — [x]; `←` — [ ] rare, keep?

## 4. Accents — no new structure needed

`\hat{x}`, `\bar{x}`, `\vec{v}`, `\dot{x}`, `\tilde{x}`: a mark *above* a
symbol. Proposal: the base symbol gets the mark as an **UPPER** child (the
edge ∑ already uses for its upper limit), and the mark is an existing
glyph in that position:

| Accent | Mark glyph | New class? |
|---|---|---|
| `\bar{x}` | `-` | no |
| `\dot{x}`, `\ddot{x}` | `.` (one / two) | no |
| `\vec{v}` | `→` | no |
| `\tilde{x}` | `~` | **yes** (`~` also = `\sim`) |
| `\hat{x}` | `^` | **yes** (`^` also = `\wedge`) |

Needs: tree-parser training data with accents (synthetic, gen_data) and
rendering. `\overline{AB}` (over several symbols) later.

- [ ] Accents via UPPER + mark glyphs, as above
- [ ] `\hat` `\bar` `\vec` `\dot` `\tilde` all in, or which ones: ______

## 5. Additions

**Tier 1 — common in high-school / university math**
- [ ] Greek lower: ζ η κ ν ξ ρ τ χ (ι υ ο: skip — look like i, u, o)
- [ ] Greek variants: ϵ/ε, φ/ϕ, ϑ (pick the forms you write)
- [ ] Greek upper: Θ Λ Ξ
- [ ] `≈ ∼ ≡ ∝`
- [ ] `⊆ ⊇ ⊃ ∉ ∅ ∖`
- [ ] `⇒ ⇔ ↦`
- [ ] `⋯` (cdots) — or treat as three `·` like `…`

**Tier 2 — less common**
- [ ] `≪ ≫`, `∓`
- [ ] `∮ ∬`
- [ ] `∠ ⊥ ∥ °`
- [ ] `↔ ↑ ↓`
- [ ] `⋮ ⋱` (matrices)
- [ ] `∧ ∨ ¬` (logic)
- [ ] `ℝ ℕ ℤ ℚ ℂ` (blackboard bold)

**Tier 3 — skip unless a user needs it**
- script/calligraphic letters (ℒ ℱ), Hebrew ℵ, `⊕ ⊗`, `†`, chemistry

## 6. Also in scope, not classes

- **Function names** (sin, log, lim…) and **text** (otherwise, if):
  regions, see the structures roadmap — not new classes.
- **Grouping of dotted symbols** (`i j : ; ! ÷`): a grouper issue, fixed
  by candidates + data, independent of this list.

## Decisions

1. Option B, glyph families + context? (§2)
2. Accents as proposed, and which ones? (§4)
3. Which additions? (§5)
4. Anything in the current 125 to drop?
