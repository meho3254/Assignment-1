"""
evaluate.py
-----------
Broader, consistent held-out evaluation of the text-only classifier.

Labels follow the ASSIGNMENT rule throughout:  rating >= 4 -> POSITIVE, else NEGATIVE
(so a 3-star review is treated as NEGATIVE). The model receives ONLY title + text
(never the rating); rating is used solely to build the answer key.

Method
------
* Split all reviews (labeled by the rule above) 70/15/15 stratified, fixed seed.
* Few-shot demonstrators are drawn from TRAIN, deduplicated and label-balanced.
* A balanced held-out sample from TEST is scored: equal POSITIVE and NEGATIVE reviews
  (a cap per class, since the data is ~9:1 positive-skewed). This is what makes the
  per-class metrics honest on imbalanced data.
* Report: per-class precision / recall / F1, a confusion matrix, balanced accuracy,
  and estimated accuracy on the natural (skewed) population.

Caching
-------
Predictions are saved to data/results/eval_assignment.csv and metrics to
data/results/eval_metrics.json. Rerun with --no-score to reuse the cache.

Usage:
    python evaluate.py              # split, few-shot, score a balanced sample (live)
    python evaluate.py --n 800      # cap per class (default 800) -> 1600 calls
    python evaluate.py --no-score   # reuse cached predictions, recompute metrics
"""
import argparse, csv, gzip, json, os, random, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from classifier import classify_one, parse_label, clean_field

SRC      = "Gift_Cards.jsonl.gz"
OUT_DIR  = "data"
RESULTS  = f"{OUT_DIR}/results"
SEED     = 6418
MISSING  = "[no title or text provided]"

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon gift-card reviews. "
    "Decide whether the customer's review is POSITIVE or NEGATIVE based only on the "
    "review TITLE and TEXT. Positive means the customer is satisfied, happy, or "
    "recommends the product. Negative means the customer is dissatisfied, frustrated, "
    "or warns others. Reply with exactly one word: POSITIVE or NEGATIVE. "
    "No explanations, no punctuation."
)


def correct_label(rating):
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def load_labeled():
    recs = []
    with gzip.open(SRC, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip("\r\n")
            if not line:
                continue
            r = json.loads(line)
            recs.append({
                "index": len(recs),
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "rating": r["rating"],
                "label": correct_label(r["rating"]),
            })
    return recs


def review_block(title, text):
    t, x = clean_field(title, 200), clean_field(text, 500)
    if not t: t = MISSING
    if not x: x = MISSING
    return f"Title: {t}\nText: {x}\nLabel:"


def dedupe(recs, key):
    seen, out = set(), []
    for r in recs:
        k = re.sub(r"\s+", " ", (r[key] or "").strip().lower())
        if k and k not in seen:
            seen.add(k); out.append(r)
    return out


def build_few_shot(pool, shots_per_class, rng):
    pos = [r for r in pool if r["label"] == "POSITIVE"]
    neg = [r for r in pool if r["label"] == "NEGATIVE"]
    # de-duplicate and prefer moderately informative reviews
    pos = [r for r in dedupe(pos, "text")
           if 40 <= len(r["text"]) + len(r["title"]) <= 240]
    neg = [r for r in dedupe(neg, "text")
           if 40 <= len(r["text"]) + len(r["title"]) <= 240]
    picked = []
    for group in (pos, neg):
        rng.shuffle(group)
        picked += rng.sample(group, min(shots_per_class, len(group)))
    demos = [f"Review:\n{review_block(r['title'], r['text'])} {r['label']}\n"
             for r in picked]
    return "\n".join(demos)


def messages_for(few_shot, rec):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            few_shot + f"\nNow classify this review:\nReview:\n{review_block(rec['title'], rec['text'])}"},
    ]


def split_stratified(recs, p_train=0.70, p_val=0.15, seed=SEED):
    rng = random.Random(seed)
    pos = [r for r in recs if r["label"] == "POSITIVE"]
    neg = [r for r in recs if r["label"] == "NEGATIVE"]
    for g in (pos, neg):
        rng.shuffle(g)
    def cut(g, *props):
        # simple sequential split by proportions
        n = len(g); edges = [0]
        s = 0
        for p in props[:-1]:
            s += int(n * p); edges.append(s)
        edges.append(n)
        return [g[edges[i]:edges[i+1]] for i in range(len(props))]
    tr_p, va_p, te_p = cut(pos, p_train, p_val, 1 - p_train - p_val)
    tr_n, va_n, te_n = cut(neg, p_train, p_val, 1 - p_train - p_val)
    train = tr_p + tr_n; val = va_p + va_n; test = te_p + te_n
    for g in (train, val, test): rng.shuffle(g)
    return train, val, test


def score(batch, few_shot, workers, progress=None):
    def job(rec):
        raw = classify_one(messages_for(few_shot, rec))
        return {"index": rec["index"], "title": rec["title"], "text": rec["text"],
                "rating": rec["rating"], "label": rec["label"], "pred": parse_label(raw)}
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = list(as_completed([ex.submit(job, r) for r in batch]))
        for i, fut in enumerate(futs, 1):
            out.append(fut.result())
            if progress and i % 200 == 0:
                print(f"  {i}/{len(batch)} ({time.time()-progress[0]:.0f}s)", flush=True)
    return out


def population_prevalence():
    """Deterministic class prior over the whole file (no scoring/API needed)."""
    pos = neu = neg = 0
    with gzip.open(SRC, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip("\r\n")
            if not line:
                continue
            r = json.loads(line)
            if r["rating"] >= 4:
                pos += 1
            elif r["rating"] == 3:
                neu += 1
            else:
                neg += 1
    tot = pos + neu + neg
    return {
        "POSITIVE": round(pos / tot, 4),
        "NEUTRAL": round(neu / tot, 4),
        "NEGATIVE": round(neg / tot, 4),
    }


def metrics(results):
    total = len(results)
    classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    cm = {a: {b: 0 for b in classes} for a in classes}   # cm[true][pred]
    unparsed = 0
    for r in results:
        if r["pred"] is None:
            unparsed += 1; continue
        cm[r["label"]][r["pred"]] += 1
    per = {}
    for c in classes:
        tp = cm[c][c]; fp = sum(cm[a][c] for a in classes if a != c)
        fn = sum(cm[c][b] for b in classes if b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "support": tp + fn}
    correct = sum(1 for r in results if r["pred"] == r["label"])
    balanced_acc = correct / total if total else 0.0
    macro_f1 = sum(per[c]["f1"] for c in classes) / len(classes)
    return {
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct, "total": total, "unparsed": unparsed,
        "per_class": per, "confusion_matrix": cm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="negatives sampled from test (positives matched)")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-score", action="store_true", help="reuse cached predictions")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    pred_path = f"{RESULTS}/eval_assignment.csv"
    met_path = f"{RESULTS}/eval_metrics.json"

    if args.no_score and os.path.exists(pred_path):
        print(f"[evaluate] reusing cached {pred_path}")
        results = []
        with open(pred_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                results.append({"index": int(r["index"]), "title": r["title"],
                                "text": r["text"], "rating": int(float(r["rating"])),
                                "label": r["label"],
                                "pred": None if r["pred"] == "" else r["pred"]})
        m = metrics(results)
    else:
        recs = load_labeled()
        train, val, test = split_stratified(recs, seed=args.seed)

        rng = random.Random(args.seed)
        few_shot = build_few_shot(train, args.shots, rng)
        print("Few-shot demo:\n" + few_shot[:500].rstrip() + "\n" + "-" * 60)

        pos = [r for r in test if r["label"] == "POSITIVE"]
        neu = [r for r in test if r["label"] == "NEUTRAL"]
        neg = [r for r in test if r["label"] == "NEGATIVE"]
        rng.shuffle(pos); rng.shuffle(neu); rng.shuffle(neg)
        n = min(args.n, len(pos), len(neu), len(neg))
        batch = rng.sample(pos, n) + rng.sample(neu, n) + rng.sample(neg, n)
        rng.shuffle(batch)
        print(f"Scoring {len(batch)} balanced reviews ({n} POS / {n} NEU / {n} NEG) from the held-out test split...")
        start = time.time()
        results = score(batch, few_shot, args.workers, progress=[start])
        elapsed = time.time() - start
        print(f"  done in {elapsed:.1f}s ({len(batch)/elapsed:.1f}/s)")

        with open(pred_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "title", "text", "rating", "label", "pred"])
            for r in results:
                w.writerow([r["index"], r["title"], r["text"], r["rating"],
                            r["label"], r["pred"] or ""])

        m = metrics(results)
        m["sampled_positives"] = n_pos
        m["sampled_negatives"] = n_neg

    # natural-population estimate (uniform for live & cached paths; deterministic
    # prevalence computed from the whole file, so it never depends on which
    # split/sample a path happened to use)
    pop = population_prevalence()
    m["class_prevalence_population"] = pop
    m["estimated_natural_accuracy"] = round(
        pop["POSITIVE"] * m["per_class"]["POSITIVE"]["recall"] +
        pop["NEGATIVE"] * m["per_class"]["NEGATIVE"]["recall"], 4)

    with open(met_path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)

    # ---- report ----
    classes = ["POSITIVE", "NEGATIVE"]
    cm = m["confusion_matrix"]
    print("\n" + "=" * 62)
    print(f"HELD-OUT EVALUATION  ({m['total']} balanced reviews)")
    print("=" * 62)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"  {'':>10}{'pred POS':>10}{'pred NEG':>10}")
    for a in classes:
        print(f"  {'true ' + a:>10}{cm[a]['POSITIVE']:>10}{cm[a]['NEGATIVE']:>10}")
    print("\nPer-class metrics:")
    for a in classes:
        p = m["per_class"][a]
        print(f"  {a:<8} prec={p['precision']:.3f} rec={p['recall']:.3f} F1={p['f1']:.3f}  (n={p['support']})")
    print(f"\nBalanced accuracy : {m['balanced_accuracy']:.4f}")
    print(f"Macro-F1          : {m['macro_f1']:.4f}")
    print(f"Unparseable       : {m.get('unparsed', 0)}")
    if "estimated_natural_accuracy" in m:
        print(f"Natural-pop accuracy (weighted by class prevalence) : {m['estimated_natural_accuracy']:.4f}")
        print(f"  population prevalence -> {m['class_prevalence_population']}")
    print(f"\nSaved predictions -> {pred_path}")
    print(f"Saved metrics     -> {met_path}")
    print(f"Reproduce no-API   -> python evaluate.py --no-score")


if __name__ == "__main__":
    main()
