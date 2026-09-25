"""
run_all.py
----------
One-shot pipeline: rebuild the splits, score the first batch (live, against
the endpoint), and regenerate the single-file HTML dashboard.

Steps that talk to the model (score_batch) need the endpoint to be reachable.
The Streamlit dashboard is a separate, long-running step - start it with the
launcher (double-click `Run Dashboard.bat`) or:
    .venv\\Scripts\\streamlit run app.py

Usage:
    python run_all.py            # full pipeline (rebuild + rescore + HTML)
    python run_all.py --no-score # skip the live model calls, just rebuild HTML
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(step, args, check=True):
    print(f"\n--- {step} ---")
    full = [sys.executable, os.path.join(ROOT, f"{step}.py"), *args]
    p = subprocess.run(full, cwd=ROOT)
    if check and p.returncode != 0:
        print(f"[run_all] {step} FAILED (exit {p.returncode})")
        sys.exit(p.returncode)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-score", action="store_true",
                    help="skip all live model calls; reuse existing scores to rebuild the dashboards")
    args = ap.parse_args()

    run("data_prep", [])
    if args.no_score:
        print("\n[run_all] --no-score: reusing data/results/batch_100_scores.csv")
    else:
        run("score_batch", [])
    # broader held-out evaluation (cache its predictions)
    if args.no_score:
        run("evaluate", ["--no-score"])
    else:
        run("evaluate", [])
    run("dashboard", [])
    run("check", [])

    print("\n" + "=" * 62)
    print(" Pipeline complete (all consistency checks passed).")
    print(" Open the dashboard:")
    print("   - Single HTML file (offline) :  double-click dashboard.html")
    print("   - Streamlit interactive app   :  double-click 'Run Dashboard.bat'")
    print("=" * 62)


if __name__ == "__main__":
    main()
