# Amazon Gift-Card Review Sentiment Classifier

Binary sentiment classifier for Amazon **Gift_Cards** reviews that uses **only the
`title` and `text` columns** — the model never sees the `rating`. It is scored against the
rating, which is kept strictly out of the model's input.

The classifier is hosted at `http://dobolyi.com:9001/v1` (OpenAI-style API, key `6418`,
model `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`). Prompting runs in few-shot mode; Qwen reasoning
is disabled so each review returns a clean `POSITIVE` / `NEGATIVE` token.

---

## Label rules
- **Scoring (answer key, NEVER shown to the model):** `rating >= 4 → POSITIVE`, else `NEGATIVE`.
- A separate experimental split used `{4,5}=POS, {1,2}=NEG`, dropping neutral `3` (see `data/`).

## Pipeline at a glance
```
Gift_Cards.jsonl.gz
   └─ data_prep.py     → data/{train,val,test,labeled_reviews}.csv  (stratified 70/15/15)
   └─ score_batch.py   → first 100 rows → data/batch_100_input.csv (title+text only),
                        data/batch_100_answerkey.csv (rating, kept separate),
                        data/results/batch_100_scores.csv (joined, for inspection)
   └─ evaluate.py      → broader held-out eval → data/results/eval_assignment.csv,
                        data/results/eval_metrics.json (per-class PRF + confusion matrix)
   └─ dashboard.py     → dashboard.html (single self-contained, offline)
   └─ app.py           → Streamlit interactive dashboard (run via launcher)
   └─ check.py         → consistency/QA check across every artifact
```

## Results
**First 100 rows of the file:** **98 / 100 correct (98.0%)**, 0 unparseable.
Positive (≥4★) 92/93 · Negative (<4★) 6/7. (Headline reads high because the first
100 rows are naturally positive-skewed — 91 of 100 are ★5.)

**Broader held-out evaluation** (2,000 balanced reviews: 1,000 POS + 1,000 NEG,
scored under the same rule — harder, because it's balanced across ratings and
includes the messy mid-ratings):

| Class | Precision | Recall | F1 | n |
|----|----|----|----|----|
| POSITIVE | 0.938 | 0.965 | 0.951 | 1000 |
| NEGATIVE | 0.964 | 0.936 | 0.950 | 1000 |

- **Balanced accuracy = 0.9505** · **Macro-F1 = 0.9505** · 0 unparseable.
- Confusion: true POS → 965 POS / 35 NEG; true NEG → 64 POS / 936 NEG.
- Estimated accuracy on the natural (9:1 positive) population: **0.962**.
- Majority-baseline macro-F1 on balanced data is 0.50, so this is a decisive gain,
  and — importantly for this task — NEGATIVE recall is 0.94 (the model actually
  *catches* the negative reviews, which the skew makes the hard thing).

## Dashboards
1. **`dashboard.html`** — single self-contained file, works offline, no server. Double-click to open.
   Recolorable theme via CSS variables (`:root`).
2. **`app.py`** (Streamlit) — interactive; filterable review table, charts, dark mode.
   - Run via launcher: **double-click `Run Dashboard.bat`** (opens `http://localhost:8501`).
   - Or: `.venv\Scripts\streamlit run app.py`
   Theme in `.streamlit/config.toml`.

## Reproduce
```bash
# 1. environment
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe streamlit        # app.py (bundles pandas/altair)
# requests is needed for score_batch/classifier (already present in the base python)

# 2. full pipeline (rebuild splits + re-score live + evaluate + rebuild dashboards + check)
python run_all.py

# 3. Streamlit dashboard
Run Dashboard.bat        # or: .venv\Scripts\streamlit run app.py
```
Use `python run_all.py --no-score` to rebuild dashboards from existing scores
(no live model calls); `python check.py` checks consistency any time.

## Scripts
| File | Purpose |
|------|---------|
| `data_prep.py` | Build 70/15/15 splits under the assignment label rule (≥4/<4). |
| `classifier.py` | Full evaluation harness (balanced samples, per-class metrics). |
| `score_batch.py` | Step-2: score the first 100 rows against the rating (≥4/<4). |
| `evaluate.py` | Broader held-out evaluation (per-class PRF + confusion matrix), cached. |
| `dashboard.py` | Emit the self-contained `dashboard.html` from the scores CSV. |
| `app.py` | Streamlit dashboard (interactive). |
| `check.py` | Consistency/QA: dashboard numbers must match the scored data. |
| `run_all.py` | One-shot pipeline orchestrator. |
| `Run Dashboard.bat` | Double-click launcher for the Streamlit app. |

## Key files
`data/batch_100_input.csv` (model input, no rating) · `data/batch_100_answerkey.csv`
(answer key, rating only) · `data/results/batch_100_scores.csv` (joined scores) ·
`data/results/report_*.json` (machine-readable metrics).
