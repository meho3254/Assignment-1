"""
data_prep.py
------------
Prepare the Amazon Gift_Cards review data for a three-class sentiment classifier.

Labels follow the ASSIGNMENT rule (derived from 'rating' for GROUND TRUTH only):
    POSITIVE  -> rating >= 4
    NEUTRAL   -> rating == 3
    NEGATIVE  -> rating <= 2

Ground-truth labels are stored separately and are NEVER shown to the classifier
model. The model is only ever given title + text.

Outputs (cached in ./data/):
    labeled_reviews.csv   all reviews with their true label
    train.csv / val.csv / test.csv   stratified 70/15/15 split

The seed is fixed for reproducibility.
"""
import gzip, json, random, os

SRC     = "Gift_Cards.jsonl.gz"
OUT_DIR = "data"
SEED    = 6418
os.makedirs(OUT_DIR, exist_ok=True)

def load_labeled():
    """Yield dicts {index, asin, title, text, label} for neutral-free reviews."""
    recs = []
    with gzip.open(SRC, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip("\r\n")
            if not line:
                continue
            r = json.loads(line)
            rating = r["rating"]
            if rating >= 4:
                label = "POSITIVE"
            elif rating == 3:
                label = "NEUTRAL"
            else:
                label = "NEGATIVE"
            recs.append({
                "index": len(recs),
                "asin": r["asin"],
                "user_id": r.get("user_id", ""),
                "timestamp": r.get("timestamp", ""),
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "label": label,
            })
    return recs

def write_csv(path, rows, cols):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: rows[i][c] for c in cols} for i in range(len(rows)))

def main():
    recs = load_labeled()
    print(f"Labeled reviews (rule: rating>=4 -> POSITIVE, <4 -> NEGATIVE): {len(recs)}")
    from collections import Counter
    print("Class distribution:", dict(Counter(r["label"] for r in recs)))

    # Compute the label feature (as a class-preserving tag) for stratification:
    # use the label itself plus a coarse product key to keep splits balanced per class.
    pos = [r for r in recs if r["label"] == "POSITIVE"]
    neu = [r for r in recs if r["label"] == "NEUTRAL"]
    neg = [r for r in recs if r["label"] == "NEGATIVE"]

    rng = random.Random(SEED)
    rng.shuffle(pos); rng.shuffle(neu); rng.shuffle(neg)

    def split_by_proportions(items, p_train, p_val, p_test):
        n = len(items)
        n_train = int(n * p_train); n_val = int(n * p_val)
        # remainder -> test
        return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

    tr_pos, va_pos, te_pos = split_by_proportions(pos, 0.70, 0.15, 0.15)
    tr_neu, va_neu, te_neu = split_by_proportions(neu, 0.70, 0.15, 0.15)
    tr_neg, va_neg, te_neg = split_by_proportions(neg, 0.70, 0.15, 0.15)

    train = tr_pos + tr_neu + tr_neg
    val   = va_pos + va_neu + va_neg
    test  = te_pos + te_neu + te_neg
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    # Persist only what the model + scorer need (never store rating).
    cols = ["index", "asin", "title", "text", "label"]

    # Full labeled dump
    all_recs = pos + neu + neg
    rng.shuffle(all_recs)
    write_csv(f"{OUT_DIR}/labeled_reviews.csv", all_recs, cols)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        write_csv(f"{OUT_DIR}/{name}.csv", split, cols)
        from collections import Counter
        print(f"{name:5s}: {len(split):6d}  {dict(Counter(r['label'] for r in split))}")

    # Sanity: save the neutral drop count note
    with open(f"{OUT_DIR}/README.txt", "w") as fh:
        fh.write("Labels (assignment rule): POSITIVE = rating>=4, NEUTRAL = rating==3, NEGATIVE = rating<=2.\n")
        fh.write(f"Seed {SEED}. Split 70/15/15 stratified by class.\n")
    print("Wrote files to", OUT_DIR, "/")

if __name__ == "__main__":
    main()
