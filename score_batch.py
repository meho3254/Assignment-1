"""
score_batch.py
--------------
Assignment Step 2 — Score a 100-row first batch against the rating.

Rule (from the assignment):  POSITIVE = rating >= 4, NEUTRAL = rating == 3, NEGATIVE = rating <= 2.
The model is given ONLY the title and text (never the rating). The rating is
held in a separate answer key and used strictly AFTER scoring to judge the model.

Outputs (in data/):
    batch_100_input.csv      model input  -> index, title, text      (NO rating)
    batch_100_answerkey.csv  answer key   -> index, rating, correct_label
    results/batch_100_scores.csv         -> index, title, text, rating,
                                            correct_label, predicted_label, raw
    Terminal: summaries for the whole batch and per rating value.

Usage:
    python score_batch.py [--n 100] [--per_rating 20] [--workers 12] [--seed 6418]
"""
import argparse, csv, gzip, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from classifier import classify_one, parse_label, clean_field  # reuse API helper + parser

SRC      = "Gift_Cards.jsonl.gz"
OUT_DIR  = "data"
SEED     = 6418
MISSING  = "[no title or text provided]"

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon gift-card reviews. "
    "Decide whether the customer's review is POSITIVE, NEUTRAL, or NEGATIVE based only on the "
    "review TITLE and TEXT. Positive means the customer is satisfied, happy, or "
    "recommends the product. Neutral means the customer's feelings are mixed or neither "
    "strongly positive nor negative. Negative means the customer is dissatisfied, frustrated, "
    "or warns others. Reply with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE. "
    "No explanations, no punctuation."
)


def load_raw():
    """All reviews with their rating (kept out of the model input)."""
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
            })
    return recs


def correct_label(rating):
    """Assignment rule: >=4 POSITIVE, ==3 NEUTRAL, <=2 NEGATIVE."""
    if rating >= 4:
        return "POSITIVE"
    elif rating == 3:
        return "NEUTRAL"
    else:
        return "NEGATIVE"


def review_block(title, text):
    t, x = clean_field(title, 200), clean_field(text, 500)
    if not t: t = MISSING
    if not x: x = MISSING
    return f"Title: {t}\nText: {x}\nLabel:"


def build_few_shot(pool, shots_per_class, rng):
    pos = [r for r in pool if correct_label(r["rating"]) == "POSITIVE"]
    neu = [r for r in pool if correct_label(r["rating"]) == "NEUTRAL"]
    neg = [r for r in pool if correct_label(r["rating"]) == "NEGATIVE"]
    def pick(lst, n):
        lst = [r for r in lst if 40 <= len(r["text"]) + len(r["title"]) <= 250]
        rng.shuffle(lst)
        return rng.sample(lst, min(n, len(lst)))
    demos = []
    for r in pick(pos, shots_per_class) + pick(neu, shots_per_class) + pick(neg, shots_per_class):
        demos.append(f"Review:\n{review_block(r['title'], r['text'])} {correct_label(r['rating'])}\n")
    return "\n".join(demos)


def messages_for(few_shot, rec):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            few_shot + f"\nNow classify this review:\nReview:\n{review_block(rec['title'], rec['text'])}"},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)   # first N rows of the file
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    recs = load_raw()
    rng = random.Random(args.seed)

    # ----- "first batch" = the literal first rows of the file -----
    batch = recs[:args.n]
    batch_ids = {r["index"] for r in batch}
    pool = [r for r in recs if r["index"] not in batch_ids]

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)
    few_shot = build_few_shot(pool, args.shots, rng)

    # ----- write model INPUT (no rating) and ANSWER KEY (with rating) -----
    input_path  = f"{OUT_DIR}/batch_100_input.csv"
    key_path    = f"{OUT_DIR}/batch_100_answerkey.csv"
    with open(input_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["index", "title", "text"])
        w.writerows([[r["index"], r["title"], r["text"]] for r in batch])
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["index", "rating", "correct_label"])
        w.writerows([[r["index"], r["rating"], correct_label(r["rating"])] for r in batch])

    print(f"First batch of the FILE: {len(batch)} reviews (rows 1-{len(batch)}).")
    print(f"Model input (no rating)   -> {input_path}")
    print(f"Answer key (rating only)  -> {key_path}")

    # ----- score: prompt contains only title+text -----
    def job(rec):
        raw = classify_one(messages_for(few_shot, rec))
        return {"index": rec["index"], "title": rec["title"], "text": rec["text"],
                "rating": rec["rating"],
                "correct": correct_label(rec["rating"]),
                "pred": parse_label(raw), "raw": raw}
    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, r): r for r in batch}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0:
                print(f"  scored {i}/{len(batch)}  ({time.time()-start:.0f}s)", flush=True)
    results.sort(key=lambda r: r["index"])
    elapsed = time.time() - start

    # ----- score against the answer key -----
    correct = sum(1 for r in results if r["pred"] == r["correct"])
    unparsed = [r for r in results if r["pred"] is None]
    accuracy = correct / len(results)

    from collections import Counter
    rating_acc = {}
    for lv in range(1, 6):
        sub = [r for r in results if int(r["rating"]) == lv]
        rating_acc[lv] = (sum(1 for r in sub if r["pred"] == r["correct"]), len(sub))

    # save scores (this file DOES include rating, for your inspection after scoring)
    os.makedirs(f"{OUT_DIR}/results", exist_ok=True)
    score_path = f"{OUT_DIR}/results/batch_100_scores.csv"
    with open(score_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "title", "text", "rating", "correct_label", "predicted_label", "raw_output"])
        for r in results:
            w.writerow([r["index"], r["title"], r["text"], r["rating"], r["correct"], r["pred"], r["raw"]])

    # ----- report -----
    print("\n" + "=" * 68)
    print(f"SCORED {len(results)} REVIEWS IN {elapsed:.1f}s  |  accuracy = {accuracy:.3f}  ({correct}/{len(results)})")
    print("=" * 68)
    print(f"Correct label rule: rating >= 4 -> POSITIVE, rating == 3 -> NEUTRAL, rating <= 2 -> NEGATIVE")
    print(f"Unparseable model outputs: {len(unparsed)}")
    print("\nAccuracy by rating value:")
    for lv in range(1, 6):
        c, n = rating_acc[lv]
        print(f"  rating {lv}: {c}/{n}  ({c/n:.0%})")
    print("\nAgreement per outcome:")
    print(f"  rated POSITIVE(>=4) correctly: "
          f"{sum(1 for r in results if r['correct']=='POSITIVE' and r['pred']=='POSITIVE')}/"
          f"{sum(1 for r in results if r['correct']=='POSITIVE')}")
    print(f"  rated NEUTRAL(==3)   correctly: "
          f"{sum(1 for r in results if r['correct']=='NEUTRAL' and r['pred']=='NEUTRAL')}/"
          f"{sum(1 for r in results if r['correct']=='NEUTRAL')}")
    print(f"  rated NEGATIVE(<=2)  correctly: "
          f"{sum(1 for r in results if r['correct']=='NEGATIVE' and r['pred']=='NEGATIVE')}/"
          f"{sum(1 for r in results if r['correct']=='NEGATIVE')}")
    print(f"\nMisclassifications: {len(results)-correct}")
    for r in results:
        if r["pred"] != r["correct"]:
            rating_val = r.get("rating") or "?"
            true_label = str(r.get("correct") or "UNKNOWN")
            pred_label = str(r.get("pred") or "UNPARSED")
            print(f"  [{rating_val}★ TRUE={true_label:<8} PRED={pred_label:<10}] "
                  f"{str(r.get('title') or '(no title)')[:40]!r} | {str(r.get('text') or '')[:70]!r}")
    print(f"\nSaved full scores (with title/text/rating) -> {score_path}")
    print(f"Saved model input (title/text only)        -> {input_path}")
    print(f"Saved answer key (rating only)             -> {key_path}")


if __name__ == "__main__":
    main()
