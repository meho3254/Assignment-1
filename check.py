"""
check.py
--------
Consistency / QA check for the pipeline outputs. Verifies that:
  1. All expected artifact files exist.
  2. The 100-row batch score (batch_100_scores.csv) is internally consistent
     and matches what dashboard.html claims.
  3. The broader evaluation metrics (eval_metrics.json) are recomputable from
     the stored predictions (eval_assignment.csv).
Exits non-zero if anything fails. Called by run_all.py at the end.

Usage:  python check.py
"""
import csv, json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(ROOT, "data", *p)

FAILS = []


def check(cond, msg):
    status = "ok " if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        FAILS.append(msg)
    return cond


def main():
    print("=== Consistency / QA check ===\n")

    # 1) artifacts exist
    expected = [
        "batch_100_input.csv", "batch_100_answerkey.csv",
        "results/batch_100_scores.csv",
        "results/eval_assignment.csv", "results/eval_metrics.json",
    ]
    for rel in expected:
        check(os.path.exists(D(rel)), f"artifact exists: {rel}")

    # 2) 100-row batch: recompute aggregate from CSV
    if os.path.exists(D("results/batch_100_scores.csv")):
        rows = list(csv.DictReader(open(D("results/batch_100_scores.csv"),
                                        newline="", encoding="utf-8")))
        if rows:
            check(len(rows) == 100, f"batch has {len(rows)} rows (expect 100)")
            correct = sum(1 for r in rows if r["correct_label"] == r["predicted_label"])
            unparsed = sum(1 for r in rows if not r["predicted_label"])
            acc = correct / len(rows)
            check(correct == 98, f"batch correct = {correct} (expect 98)")
            check(unparsed == 0, f"batch unparsed = {unparsed} (expect 0)")
            # dashboard.html claims the same
            if os.path.exists(os.path.join(ROOT, "dashboard.html")):
                html = open(os.path.join(ROOT, "dashboard.html"), encoding="utf-8").read()
                claim = f"{acc*100:.1f}%" if acc < 0.995 else "100%"
                check(claim in html,
                      f"dashboard.html headline claims {claim}")
                # embedded JSON equals the CSV
                m = re.search(r'<script type="application/json" id="data">(.*?)</script>',
                              html, re.S)
                if m:
                    data = json.loads(m.group(1))
                    check(len(data) == len(rows), "dashboard embeds every batch review")

    # 3) broader eval metrics recompute from predictions
    if os.path.exists(D("results/eval_assignment.csv")) and \
       os.path.exists(D("results/eval_metrics.json")):
        preds = list(csv.DictReader(open(D("results/eval_assignment.csv"),
                                         newline="", encoding="utf-8")))
        met = json.load(open(D("results/eval_metrics.json"), encoding="utf-8"))
        correct = sum(1 for r in preds if r["label"] == r["pred"])
        nn = len(preds)
        check(nn == met.get("total"), f"eval rows {nn} match metrics total {met.get('total')}")
        bal = round(correct / nn, 4) if nn else 0
        check(abs(bal - met.get("balanced_accuracy", -1)) < 1e-9,
              f"recomputed balanced accuracy {bal} == stored {met.get('balanced_accuracy')}")
        check(met.get("unparsed", -1) == 0, "eval has 0 unparseable")

    print()
    if FAILS:
        print(f"=== {len(FAILS)} CHECK(S) FAILED ===")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
