"""
classifier.py
-------------
Classify Amazon Gift_Cards reviews as POSITIVE / NEGATIVE using ONLY the
title and text columns via a hosted LLM (OpenAI-compatible vLLM endpoint).

The model never sees 'rating' (or any label) - it is a zero/few-shot
sentiment classifier driven purely by the review text. Ground-truth labels
(derived from rating) are used only to score the output.

Usage:
    python classifier.py --split val   --n_pos 120 --n_neg 120
    python classifier.py --split test  --n_pos 250 --n_neg 250

Flags:
    --split         'val' or 'test' split to evaluate
    --n_pos/--n_neg how many POSITIVE / NEGATIVE reviews to sample (balanced)
    --shots         number of few-shot demonstrations PER class from train
    --workers       concurrent requests (default 10)
    --probe         if set, classify just `--probe` reviews and print them
    --seed          RNG seed for sampling (default 6418)
"""
import argparse, csv, json, os, random, re, sys, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://dobolyi.com:9001/v1"
API_KEY  = "6418"
MODEL    = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
DATA_DIR = "data"
SEED     = 6418

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon gift-card reviews. "
    "Decide whether the customer's review is POSITIVE, NEUTRAL, or NEGATIVE based only on "
    "the review TITLE and TEXT. "
    "Positive means the customer is satisfied, happy, or recommends the product. "
    "Neutral means the customer's feelings are mixed or neither strongly positive nor negative. "
    "Negative means the customer is dissatisfied, frustrated, or warns others. "
    "Reply with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE. No explanations, no punctuation."
)

MISSING = "[no " + "title or text provided]"


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def read_csv(path):
    recs = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            recs.append(row)
    return recs


def clean_field(s, limit=300):
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)          # collapse whitespace/newlines
    return s[:limit]


def review_block(title, text):
    t = clean_field(title, 200)
    x = clean_field(text, 500)
    if not t:
        t = MISSING
    if not x:
        x = MISSING
    return f"Title: {t}\nText: {x}\nLabel:"


# --------------------------------------------------------------------------
# Few-shot demonstrations (from TRAIN - never from the eval split)
# --------------------------------------------------------------------------
def build_few_shot(train_recs, shots_per_class):
    pos = [r for r in train_recs if r["label"] == "POSITIVE"]
    neg = [r for r in train_recs if r["label"] == "NEGATIVE"]
    rng = random.Random(SEED)
    # pick reasonably informative, shorter examples
    def pick(pool, n, pref):
        pool = [r for r in pool if 40 <= len(r.get("text", "")) + len(r.get("title", "")) <= 250]
        pool = sorted(pool, key=lambda r: r.get("text", "").count(" ") + r.get("title", "").count(" "))
        if n == 0:
            return []
        rng.shuffle(pool)
        chosen = rng.sample(pool if len(pool) <= n else pool[:500], min(n, len(pool), 500))
        return chosen

    demos = []
    for r in (pick(pos, shots_per_class, "POS") + pick(neg, shots_per_class, "NEG")):
        lbl = r["label"]
        demos.append(f"Review:\n{review_block(r['title'], r['text'])} {lbl}\n")
    return "\n".join(demos)

def build_messages(few_shot_blocks, target):
    user = ()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": few_shot_blocks + f"\nNow classify this review:\nReview:\n{review_block(target['title'], target['text'])}"},
    ]


# --------------------------------------------------------------------------
# API call
# --------------------------------------------------------------------------
def classify_one(messages, timeout=60):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 8,
        "temperature": 0,
        # Qwen3 is a reasoning model; disable chain-of-thought so we get a
        # clean answer token instead of a streamed thinking block.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", json=payload,
                              headers=headers, timeout=timeout)
            if r.status_code == 200:
                j = r.json()
                content = j["choices"][0]["message"].get("content") or ""
                return content.strip()
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    return f"__ERR__{last_err}"


def parse_label(raw):
    """Robustly map model output to POSITIVE/NEGATIVE."""
    if not raw:
        return None
    up = raw.upper()
    if "__ERR__" in up:
        return None
    if "POSITIVE" in up and "NEGATIVE" not in up:
        return "POSITIVE"
    if "NEGATIVE" in up and "POSITIVE" not in up:
        return "NEGATIVE"
    # Co-occurrence or weird output -> conservative word-boundary check
    m = re.search(r"\b(POSITIVE|NEGATIVE)\b", up)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(predictions):
    """predictions: list of dicts with 'true' and 'pred' (both in POSITIVE/NEGATIVE/None)."""
    classes = ["POSITIVE", "NEGATIVE"]
    tp, fp, fn = {}, {}, {}
    for c in classes:
        tp[c] = sum(1 for p in predictions if p["pred"] == c and p["true"] == c)
        fp[c] = sum(1 for p in predictions if p["pred"] == c and p["true"] != c)
        fn[c] = sum(1 for p in predictions if p["true"] == c and p["pred"] != c)

    rows = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                   "support": tp[c] + fn[c]}
    correct = sum(1 for p in predictions if p["pred"] == p["true"])
    total   = len(predictions)
    unparsed = sum(1 for p in predictions if p["pred"] is None)
    accuracy = correct / total if total else 0.0
    macro_f1 = sum(rows[c]["f1"] for c in classes) / len(classes)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "unparsed": unparsed,
        "per_class": rows,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def sample_split(split_recs, n_pos, n_neg, seed):
    pos = [r for r in split_recs if r["label"] == "POSITIVE"]
    neg = [r for r in split_recs if r["label"] == "NEGATIVE"]
    rng = random.Random(seed)
    pos = rng.sample(pos, min(n_pos, len(pos)))
    neg = rng.sample(neg, min(n_neg, len(neg)))
    chosen = pos + neg
    rng.shuffle(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--n_pos", type=int, default=250)
    ap.add_argument("--n_neg", type=int, default=250)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    train = read_csv(f"{DATA_DIR}/train.csv")
    split = read_csv(f"{DATA_DIR}/{args.split}.csv")

    few_shot = build_few_shot(train, args.shots)
    print("Few-shot demo (first 600 chars):\n", few_shot[:600].rstrip(), "\n" + "-" * 70)

    target = sample_split(split, args.n_pos, args.n_neg, args.seed)
    if args.probe:
        target = target[:args.probe]

    print(f"Evaluating {len(target)} reviews on '{args.split}' split "
          f"with {args.shots} shots/class, {args.workers} workers...")

    def job(rec):
        msgs = build_messages(few_shot, rec)
        raw = classify_one(msgs)
        pred = parse_label(raw)
        return {"index": rec["index"], "asin": rec["asin"], "true": rec["label"],
                "pred": pred, "raw": raw, "title": rec["title"], "text": rec["text"]}

    predictions = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, rec): rec for rec in target}
        done = 0
        for fut in as_completed(futs):
            predictions.append(fut.result())
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(target)} done  ({time.time()-start:.0f}s)", flush=True)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(target)/elapsed:.2f} reviews/s)")

    metrics = compute_metrics(predictions)
    print(json.dumps(metrics, indent=2))

    # Save outputs
    tag = f"{args.split}_s{args.shots}_{args.seed}"
    out_dir = os.path.join(DATA_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, f"predictions_{tag}.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "asin", "true_label", "pred_label", "raw_output",
                    "title", "text"])
        for p in predictions:
            w.writerow([p["index"], p["asin"], p["true"], p["pred"], p["raw"],
                        p.get("title", ""), p.get("text", "")])

    meta = {"split": args.split, "n_pos": args.n_pos, "n_neg": args.n_neg,
            "shots": args.shots, "seed": args.seed, "review_count": len(target),
            "elapsed_s": round(elapsed, 1), "metrics": metrics,
            "predictions_csv": pred_path}
    report_path = os.path.join(out_dir, f"report_{tag}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved:", pred_path)
    print("Saved:", report_path)

    if args.probe:
        print("\n--- PROBE DETAIL ---")
        for p in predictions:
            print(f"true={p['true']:<8} pred={p['pred']:<8} | {p['raw'][:60]!r}")


if __name__ == "__main__":
    main()
