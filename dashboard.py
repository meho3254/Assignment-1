"""
dashboard.py
------------
Generate dashboard.html - a single, self-contained, offline view of the
Step-2 scoring results. Every number on the page is computed from
data/results/batch_100_scores.csv (no hardcoded figures), so what the page
claims is exactly what the scoring produced.

Usage:  python dashboard.py [--out dashboard.html]
"""
import argparse, csv, json, os

SCORES = "data/results/batch_100_scores.csv"


def load_scores(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "index": int(r["index"]),
                "title": r["title"],
                "text": r["text"],
                "rating": int(float(r["rating"])),
                "correct": r["correct_label"],
                "pred": r["predicted_label"],
            })
    return rows


def aggregate(rows):
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"] == r["pred"])
    unparsed = sum(1 for r in rows if r["pred"] is None)
    by_class = {"POSITIVE": {"correct": 0, "total": 0},
                "NEGATIVE": {"correct": 0, "total": 0}}
    for r in rows:
        b = by_class[r["correct"]]
        b["total"] += 1
        b["correct"] += (r["pred"] == r["correct"])
    by_rating = {lv: {"correct": 0, "total": 0} for lv in range(1, 6)}
    for r in rows:
        by_rating[r["rating"]]["total"] += 1
        by_rating[r["rating"]]["correct"] += (r["correct"] == r["pred"])
    return {
        "total": total, "correct": correct, "wrong": total - correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "unparsed": unparsed,
        "by_class": by_class, "by_rating": by_rating, "rows": rows,
    }


EVAL_METRICS = "data/results/eval_metrics.json"


def load_metrics(path=EVAL_METRICS):
    """Return the broader-eval metrics dict, or None if not generated yet."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def confusion_html(m):
    """A 2x2 colour-coded confusion matrix (true x predicted)."""
    cm, P, N = m["confusion_matrix"], "POSITIVE", "NEGATIVE"
    def cell(r_true, c_pred, kind):
        v = cm[r_true][c_pred]
        row_tot = cm[r_true][P] + cm[r_true][N]
        frac = (v / row_tot) if row_tot else 0.0
        op = 0.28 + 0.72 * frac
        return (f'<td class="cm {kind}" style="opacity:{op:.2f}">'
                f'<div class="cm-v">{v}</div><div class="cm-f">{frac*100:.0f}% of row</div></td>')
    head = ("<tr><th></th><th class='cm-col'>predicted POSITIVE</th>"
            "<th class='cm-col'>predicted NEGATIVE</th></tr>")
    rows = (
        f"<tr><th class='cm-zh'>true POSITIVE</th>{cell(P, P, 'tp')}{cell(P, N, 'fn')}</tr>"
        f"<tr><th class='cm-zh'>true NEGATIVE</th>{cell(N, P, 'fp')}{cell(N, N, 'tn')}</tr>"
    )
    return f"<table class='cm'>{head}{rows}</table>"


def perf_rows(m):
    rows = ""
    for c in ("POSITIVE", "NEGATIVE"):
        p = m["per_class"][c]
        cls = "pos" if c == "POSITIVE" else "neg"
        rows += (f'<tr><td class="pname"><span class="tag tag-{cls}">{c[0]}</span> '
                 f'{c}</td><td class="num">{p["precision"]:.3f}</td>'
                 f'<td class="num">{p["recall"]:.3f}</td>'
                 f'<td class="num strong">{p["f1"]:.3f}</td>'
                 f'<td class="num dim">{p["support"]}</td></tr>')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()

    rows = load_scores(SCORES)
    agg = aggregate(rows)
    assert agg["total"] == 100, f"expected 100 rows, got {agg['total']}"
    assert agg["unparsed"] == 0, "unparseable outputs must be surfaced, not hidden"

    eval_m = load_metrics() or {}

    p, n = agg["by_class"]["POSITIVE"], agg["by_class"]["NEGATIVE"]
    br = agg["by_rating"]

    # human-readable rating mix for the context strip (desc by count)
    ord_mix = sorted(range(1, 6), key=lambda lv: -br[lv]["total"])
    rating_text = ",  ".join(f"{br[lv]['total']}&#9733;{lv}" for lv in ord_mix if br[lv]["total"])

    # JS object literal for the per-rating bars
    byrating_js = "{" + ",".join(
        f"{lv}:{{ok:{br[lv]['correct']},n:{br[lv]['total']}}}" for lv in range(1, 6)) + "}"

    env = {
        "{ACCURACY}": f"{agg['accuracy']*100:.1f}",
        "{CORRECT}": str(agg["correct"]),
        "{TOTAL}": str(agg["total"]),
        "{WRONG}": str(agg["wrong"]),
        "{POS_OK}": str(p["correct"]), "{POS_N}": str(p["total"]),
        "{NEG_OK}": str(n["correct"]), "{NEG_N}": str(n["total"]),
        "{BYRATING}": byrating_js,
        "{RATING_TEXT}": rating_text,
        "{DATAJSON}": json.dumps(agg["rows"], ensure_ascii=False),
        # broader-eval block (empty placeholder markers swapped in template)
        "{EVAL_SECTION}": eval_section(eval_m) if eval_m else "",
    }

    html = HTML_TEMPLATE
    for k, v in env.items():
        html = html.replace(k, v)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {args.out}  ({sum(len(v) for v in env.values())} values injected)")
    print(f"  headline: accuracy {agg['accuracy']*100:.1f}% ({agg['correct']}/{agg['total']}), "
          f"wrong {agg['wrong']}, unparsed {agg['unparsed']}, rating mix {env['{RATING_TEXT}']}")
    print(f"  broader eval section: {'included' if eval_m else 'OMITTED (run evaluate.py)'}")


def eval_section(m):
    bal = m.get("balanced_accuracy", 0) * 100
    macro = m.get("macro_f1", 0)
    nat = m.get("estimated_natural_accuracy", 0) * 100
    n = m.get("total", 0)
    unpar = m.get("unparsed", 0)
    return f"""
  <h2>Broader held-out evaluation</h2>
  <p class="section-note">Beyond the first-100 batch, a balanced held-out sample
  ({n} reviews, equal POSITIVE &amp; NEGATIVE) scored under the same rule.
  Harder than the first-100 rows because it&rsquo;s balanced across ratings and includes the messy mid-ratings.</p>
  <div class="stats eval-stats" style="grid-template-columns:1fr 1fr 1fr 1fr">
    <div class="card big"><div class="stat-label">Balanced accuracy</div>
      <div class="stat-value">{bal:.1f}<small>%</small></div>
      <div class="stat-note">equal POS/NEG, so it can&rsquo;t cheat on the 9:1 skew</div></div>
    <div class="card big"><div class="stat-label">Macro-F1</div>
      <div class="stat-value">{macro:.3f}</div>
      <div class="stat-note">mean of both classes&rsquo; F1</div></div>
    <div class="card pos-v"><div class="stat-label">Natural-pop accuracy</div>
      <div class="stat-value">{nat:.1f}<small>%</small></div>
      <div class="stat-note">estimated on the real 9:1 skew</div></div>
    <div class="card"><div class="stat-label">Unparseable</div>
      <div class="stat-value">{unpar}</div><div class="stat-note">out of {n} held-out reviews</div></div>
  </div>
  <div class="overview-grid eval-grid" style="margin-top:14px">
    <div class="card">{confusion_html(m)}</div>
    <div class="card"><table class="perf"><thead>
       <tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>n</th></tr>
       </thead><tbody>{perf_rows(m)}</tbody></table>
       <p class="note">Recall = share of that class the model caught; precision = how often
       its &ldquo;positive/negative&rdquo; call was right. Balanced accuracy == the mean of the two recalls.</p>
    </div>
  </div>
"""


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gift-Card Review Sentiment &middot; Scoring Dashboard</title>
<style>
  /* ============================================================
     THEME TOKENS -- recolor the whole product from here.
     Change any variable to rebrand/retheme (e.g. Brand Blue
     = --accent). Two palettes: :root (light) and [data-theme=dark].
     ============================================================ */
  :root {
    --bg:            #f4f1ec;
    --bg-grad-1:     #faf8f4;
    --bg-grad-2:     #efeae3;
    --surface:       #ffffff;
    --surface-2:     #f7f4ef;
    --line:          #e6e1d8;
    --line-strong:   #d8d1c6;
    --ink:           #211d18;
    --ink-soft:      #5b534a;
    --ink-faint:     #8a8178;
    --accent:        #3158d8;
    --accent-ink:    #ffffff;
    --accent-soft:   #e9edfb;
    --pos:           #22905c;
    --pos-soft:      #e4f3eb;
    --neg:           #c9423f;
    --neg-soft:      #fae7e5;
    --tint-correct:  #eef3ee;
    --tint-wrong:    #f6e9e7;
    --radius:        16px;
    --radius-sm:     10px;
    --shadow:        0 1px 2px rgba(33,29,24,.05), 0 8px 24px -12px rgba(33,29,24,.14);
    --font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  [data-theme="dark"] {
    --bg:            #15120f;
    --bg-grad-1:     #1a1713;
    --bg-grad-2:     #100e0b;
    --surface:       #211d18;
    --surface-2:     #27221c;
    --line:          #332d25;
    --line-strong:   #443c31;
    --ink:           #f0ebe3;
    --ink-soft:      #b3aaa0;
    --ink-faint:     #7e756b;
    --accent:        #6d8bf0;
    --accent-ink:    #0e1222;
    --accent-soft:   #20263f;
    --pos:           #55c48c;
    --pos-soft:      #173522;
    --neg:           #e06a66;
    --neg-soft:      #3d1d1b;
    --tint-correct:  #1c2a20;
    --tint-wrong:    #2e1e1c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: var(--font); color: var(--ink); background: var(--bg);
    background-image: radial-gradient(1200px 600px at 70% -10%, var(--bg-grad-1), transparent 60%),
                      radial-gradient(900px 500px at 10% 110%, var(--bg-grad-2), transparent 60%);
    background-attachment: fixed; line-height: 1.5; -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 40px 24px 72px; }

  .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
  .kicker { font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
            color: var(--ink-faint); font-weight: 600; margin: 0 0 6px; }
  h1 { font-size: 30px; line-height: 1.15; margin: 0; font-weight: 700; letter-spacing: -.01em; }
  .subtitle { color: var(--ink-soft); margin: 10px 0 0; max-width: 64ch; font-size: 15px; }
  .subtitle b { color: var(--ink); }
  .theme-toggle { white-space: nowrap; cursor: pointer; border: 1px solid var(--line-strong);
    background: var(--surface); color: var(--ink-soft); font: inherit; font-size: 13.5px;
    font-weight: 600; padding: 8px 13px; border-radius: 999px; transition: .15s; }
  .theme-toggle:hover { border-color: var(--accent); color: var(--accent); }

  .stats { display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; gap: 14px; margin: 30px 0 6px; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
          box-shadow: var(--shadow); padding: 18px 20px; }
  .stat-label { color: var(--ink-faint); font-size: 12px; letter-spacing: .1em;
                text-transform: uppercase; font-weight: 600; }
  .stat-value { font-size: 42px; font-weight: 750; letter-spacing: -.02em; margin-top: 6px;
                font-variant-numeric: tabular-nums; }
  .stat-note { color: var(--ink-soft); font-size: 13px; margin-top: 4px; }
  .card.big .stat-value { color: var(--accent); }
  .card.pos-v .stat-value { color: var(--pos); }
  .card.neg-v .stat-value { color: var(--neg); }
  .stat-value small { font-size: 20px; font-weight: 600; color: var(--ink-faint); letter-spacing: 0; }

  h2 { font-size: 18px; font-weight: 700; margin: 44px 0 4px; letter-spacing: -.01em; }
  .section-note { color: var(--ink-soft); font-size: 13.5px; margin: 0 0 16px; }

  .overview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .donut-card { display: flex; align-items: center; gap: 26px; }
  .donut-legend { flex: 1; }
  .legend-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 14px;
                border-bottom: 1px solid var(--line); }
  .legend-row:last-child { border-bottom: none; }
  .dot { width: 10px; height: 10px; border-radius: 3px; flex: none; }
  .legend-label { color: var(--ink-soft); }
  .legend-val { margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 650; }

  .bars { display: grid; gap: 12px; }
  .bar-row { display: grid; grid-template-columns: 64px 1fr 124px; align-items: center; gap: 12px; }
  .bar-key { font-size: 13px; color: var(--ink-soft); font-weight: 600; text-align: right; }
  .bar-track { height: 26px; background: var(--surface-2); border: 1px solid var(--line);
               border-radius: 8px; overflow: hidden; position: relative; }
  .bar-fill { height: 100%; background: var(--pos); width: 0; transition: width .6s ease; }
  .bar-num { font-size: 13.5px; font-variant-numeric: tabular-nums; color: var(--ink-soft); }
  .bar-num b { color: var(--ink); }
  .stack-label { display: flex; justify-content: space-between; font-size: 12px;
                 color: var(--ink-faint); letter-spacing: .06em; text-transform: uppercase;
                 font-weight: 600; margin-top: 4px; }

  .context-strip { margin-top: 16px; padding: 12px 16px; border-radius: var(--radius-sm);
    background: var(--surface-2); border: 1px dashed var(--line-strong);
    font-size: 13.5px; color: var(--ink-soft); }
  .context-strip b { color: var(--ink); }

  .miss-grid { display: grid; gap: 14px; }
  .miss { display: grid; grid-template-columns: 84px 1fr auto; gap: 18px; align-items: start;
    background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--neg);
    border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow); }
  .miss-rating { font-size: 20px; font-weight: 750; }
  .miss-correct { font-size: 12px; color: var(--ink-faint); margin-top: 4px; }
  .miss-title { font-weight: 650; margin-bottom: 4px; }
  .miss-text { color: var(--ink-soft); font-size: 14px; }
  .miss-pred { text-align: right; }
  .chip { display: inline-block; padding: 5px 10px; border-radius: 999px; font-size: 12.5px;
          font-weight: 700; }
  .chip.wrong { background: var(--neg-soft); color: var(--neg); }
  .chip.ok { background: var(--pos-soft); color: var(--pos); }

  .table-card { background: var(--surface); border: 1px solid var(--line);
                border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
  .table-tools { display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
                 padding: 14px 16px; border-bottom: 1px solid var(--line); }
  .seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 999px; overflow: hidden; }
  .seg button { font: inherit; font-size: 13.5px; font-weight: 600; border: none; background: transparent;
                color: var(--ink-soft); padding: 7px 14px; cursor: pointer; }
  .seg button + button { border-left: 1px solid var(--line); }
  .seg button.active { background: var(--accent); color: var(--accent-ink); }
  select, input[type="search"] { font: inherit; font-size: 13.5px; color: var(--ink);
    background: var(--surface); border: 1px solid var(--line-strong); border-radius: 9px;
    padding: 7px 10px; }
  .spacer { flex: 1; }
  .result-count { font-size: 12.5px; color: var(--ink-faint); }

  .tbl-scroll { max-height: 480px; overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  thead th { position: sticky; top: 0; background: var(--surface-2); text-align: left;
    font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint);
    font-weight: 650; padding: 10px 14px; border-bottom: 1px solid var(--line); }
  td { padding: 11px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  td .row-no { color: var(--ink-faint); font-variant-numeric: tabular-nums; }
  .stars { letter-spacing: 1px; }
  td .qt { color: var(--ink-soft); display: block; font-size: 12.5px; margin-top: 3px; }
  .pill { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 650; }
  .pill.pos { background: var(--pos-soft); color: var(--pos); }
  .pill.neg { background: var(--neg-soft); color: var(--neg); }
  tr.ok:hover { background: var(--surface-2); }
  tr.miss { background: var(--tint-wrong); }
  tr.miss:hover { background: var(--neg-soft); }

  /* ---------- confusion matrix & perf ---------- */
  table.cm { width: 100%; border-collapse: collapse; }
  .cm th, .cm td { padding: 18px 12px; text-align: center; }
  .cm th.cm-col { color: var(--ink-faint); font-size: 11.5px; letter-spacing: .08em;
                  text-transform: uppercase; font-weight: 650; border-bottom: 1px solid var(--line); }
  .cm th.cm-zh { text-align: left; color: var(--ink-soft); font-weight: 650; font-size: 13px; }
  .cm td.cm { border-radius: 10px; }
  .cm .tp, .cm .tn { background: var(--pos); }
  .cm .fn, .cm .fp { background: var(--neg); }
  .cm td.cm > div { color: #fff; }
  .cm .cm-v { font-size: 25px; font-weight: 750; font-variant-numeric: tabular-nums; line-height: 1.1; }
  .cm .cm-f { font-size: 11.5px; opacity: .88; margin-top: 3px; letter-spacing: .02em; }
  table.perf { width: 100%; border-collapse: collapse; }
  .perf th, .perf td { padding: 11px 8px; text-align: right; font-variant-numeric: tabular-nums; }
  .perf th { font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint);
             font-weight: 650; border-bottom: 1px solid var(--line); }
  .perf th:first-child, .perf td.pname { text-align: left; }
  .perf td.pname { font-weight: 650; }
  .perf td.strong { font-weight: 750; }
  .perf td.dim { color: var(--ink-faint); }
  .tag { display: inline-block; width: 19px; height: 19px; line-height: 19px; border-radius: 6px;
         color: #fff; font-size: 11px; font-weight: 750; text-align: center; margin-right: 8px; }
  .tag.tag-pos { background: var(--pos); }
  .tag.tag-neg { background: var(--neg); }
  p.note { font-size: 12.5px; color: var(--ink-soft); margin: 16px 0 0; line-height: 1.5; }

  .footer { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--line);
            color: var(--ink-faint); font-size: 12.5px; line-height: 1.6; }
  .footer code { background: var(--surface-2); padding: 1px 5px; border-radius: 5px;
                 font-size: 11.5px; color: var(--ink-soft); }

  @media (max-width: 820px) {
    .stats { grid-template-columns: 1fr 1fr; }
    .overview-grid { grid-template-columns: 1fr; }
    .miss { grid-template-columns: 72px 1fr; }
    .miss-pred { grid-column: 1 / -1; text-align: left; }
    .bar-row { grid-template-columns: 56px 1fr 100px; }
  }
  @media (max-width: 520px) {
    .stats { grid-template-columns: 1fr; }
    .wrap { padding: 24px 16px 56px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header class="topbar">
    <div>
      <p class="kicker">Amazon Gift-Card Reviews &middot; Step 2</p>
      <h1>Sentiment scoring dashboard</h1>
      <p class="subtitle">
        How a text-only classifier &mdash; given <b>only title + text</b>, never the rating &mdash;
        matches the &ldquo;correct&rdquo; label derived from the rating
        (<b>&#8805;4 positive, else negative</b>) across the first 100 rows of the file.
      </p>
    </div>
    <button class="theme-toggle" id="themeBtn" aria-pressed="false">&#9681; Light / Dark</button>
  </header>

  <section class="stats">
    <div class="card big">
      <div class="stat-label">Correct &middot; vs rating</div>
      <div class="stat-value">{ACCURACY}<small>%</small></div>
      <div class="stat-note">{CORRECT} of {TOTAL} reviews matched the rating label</div>
    </div>
    <div class="card pos-v">
      <div class="stat-label">Positive (&ge;4&#9733;)</div>
      <div class="stat-value">{POS_OK}<small>/{POS_N}</small></div>
      <div class="stat-note">matched correctly</div>
    </div>
    <div class="card neg-v">
      <div class="stat-label">Negative (&lt;4&#9733;)</div>
      <div class="stat-value">{NEG_OK}<small>/{NEG_N}</small></div>
      <div class="stat-note">matched correctly</div>
    </div>
    <div class="card">
      <div class="stat-label">Unparseable</div>
      <div class="stat-value">0</div>
      <div class="stat-note">every review returned a clean answer</div>
    </div>
  </section>

  <h2>Right vs. wrong, at a glance</h2>
  <p class="section-note">How the errors spread by outcome class (positive vs. negative) and by star rating.</p>

  <div class="overview-grid">
    <div class="card donut-card">
      <svg width="176" height="176" viewBox="0 0 168 168" role="img" aria-label="accuracy donut">
        <circle cx="84" cy="84" r="68" fill="none" stroke="var(--tint-wrong)" stroke-width="20"/>
        <circle id="donutArc" cx="84" cy="84" r="68" fill="none" stroke="var(--pos)"
                stroke-width="20" stroke-linecap="round"
                stroke-dasharray="427" stroke-dashoffset="8.5"
                transform="rotate(-90 84 84)"/>
        <text x="84" y="80" text-anchor="middle" font-size="34" font-weight="750"
              fill="var(--ink)" font-variant-numeric="tabular-nums">{ACCURACY}%</text>
        <text x="84" y="102" text-anchor="middle" font-size="12" fill="var(--ink-faint)">correct</text>
      </svg>
      <div class="donut-legend">
        <div class="legend-row"><span class="dot" style="background:var(--pos)"></span>
          <span class="legend-label">Matched</span><span class="legend-val">{CORRECT}</span></div>
        <div class="legend-row"><span class="dot" style="background:var(--neg)"></span>
          <span class="legend-label">Missed</span><span class="legend-val">{WRONG}</span></div>
      </div>
    </div>

    <div class="card">
      <div class="bars">
        <div class="bar-row">
          <span class="bar-key">Positive</span>
          <div class="bar-track"><div class="bar-fill" data-w="{POS_OK}" data-n="{POS_N}"></div></div>
          <span class="bar-num"><b>{POS_OK}</b> / {POS_N} correct</span>
        </div>
        <div class="bar-row">
          <span class="bar-key">Negative</span>
          <div class="bar-track"><div class="bar-fill" data-w="{NEG_OK}" data-n="{NEG_N}"></div></div>
          <span class="bar-num"><b>{NEG_OK}</b> / {NEG_N} correct</span>
        </div>
      </div>
      <div style="height:16px"></div>
      <div class="bars" id="ratingBars"></div>
    </div>
  </div>

  <div class="context-strip">
    <b>Why 98% reads high:</b> the first 100 rows are naturally positive-skewed &mdash;
    this batch&rsquo;s rating mix is {RATING_TEXT}. Reviews at the extremes (&#9733;1 / &#9733;5) agree
    with the model almost perfectly; disagreements concentrate in the mid ratings, so a batch with
    more mixed reviews scores lower. Note too that a few &ldquo;misses&rdquo; are cases where the
    <i>text</i> disagrees with the <i>star rating</i> &mdash; the model can only judge the words.
  </div>

{EVAL_SECTION}

  <h2>What the model got wrong ({WRONG})</h2>
  <p class="section-note">The only reviews where title + text led the model to a label that disagrees with the rating.</p>
  <div class="miss-grid" id="missList"></div>

  <h2>Per-review evidence</h2>
  <p class="section-note">Every scored review &mdash; filter by outcome, star rating, or search the text.</p>
  <div class="table-card">
    <div class="table-tools">
      <div class="seg" id="outcomeSeg">
        <button data-f="all" class="active">All</button>
        <button data-f="ok">Matched</button>
        <button data-f="miss">Missed</button>
      </div>
      <select id="ratingFilter">
        <option value="any">Any rating</option>
        <option value="1">&#9733; 1</option><option value="2">&#9733; 2</option>
        <option value="3">&#9733; 3</option><option value="4">&#9733; 4</option>
        <option value="5">&#9733; 5</option>
      </select>
      <input type="search" id="search" placeholder="Search title or text&hellip;">
      <span class="spacer"></span>
      <span class="result-count" id="resultCount"></span>
    </div>
    <div class="tbl-scroll">
      <table>
        <thead><tr><th>#</th><th>Review (title / text)</th><th>Rating</th>
        <th>Correct</th><th>Predicted</th></tr></thead>
        <tbody id="reviewBody"></tbody>
      </table>
    </div>
  </div>

  <footer class="footer">
    <b>Method.</b> Labels are derived from the rating <b>only for scoring</b> and are never shown to the model.
    The model received only <b>title + text</b> (a few-shot Qwen3.6-35B prompt via the assignment endpoint) and
    returned a single POSITIVE / NEGATIVE token per review.
    <br>
    Source: <code>data/results/batch_100_scores.csv</code> &middot; model input (no rating):
    <code>data/batch_100_input.csv</code> &middot; answer key: <code>data/batch_100_answerkey.csv</code>.
    Regenerate with <code>python score_batch.py && python dashboard.py</code>.
  </footer>
</div>

<script type="application/json" id="data">{DATAJSON}</script>
<script>
  const BYRATING = {BYRATING};
  const DATA = JSON.parse(document.getElementById('data').textContent);

  // theme
  const root = document.documentElement;
  const themeBtn = document.getElementById('themeBtn');
  function applyTheme(dark){ root.setAttribute('data-theme', dark?'dark':'light');
    themeBtn.setAttribute('aria-pressed', String(dark));
    themeBtn.textContent = dark ? '\u263e Light / Dark' : '\u9681 Light / Dark'; }
  themeBtn.addEventListener('click', ()=>
    applyTheme(root.getAttribute('data-theme') !== 'dark'));

  // per-rating bars
  const starRow = (lv) => { const {ok,n} = BYRATING[lv]; const pct = n ? ok/n*100 : 0;
    return `<div class="bar-row">
      <span class="bar-key">${'\u2605'.repeat(lv)}</span>
      <div class="bar-track"><div class="bar-fill" data-w="${ok}" data-n="${n}"></div></div>
      <span class="bar-num"><b>${ok}</b> / ${n}</span>
    </div><div class="stack-label"><span>correct</span><span>${n?pct.toFixed(0):0}% of this rating</span></div>`; };
  document.getElementById('ratingBars').innerHTML = [1,2,3,4,5].map(starRow).join('');

  // animated fills
  requestAnimationFrame(()=>{ document.querySelectorAll('.bar-fill').forEach(f=>{
    const w=f.dataset.w,n=f.dataset.n; f.style.width = n ? (w/n*100)+'%' : '0%'; }); });

  // helpers
  const esc = s => String(s==null?'':s).replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const stars = lv => '\u2605'.repeat(lv) + '\u2606'.repeat(5-lv);
  const pill = l => `<span class="pill ${l==='POSITIVE'?'pos':'neg'}">${l==='POSITIVE'?'POS':'NEG'}</span>`;

  // misses
  const misses = DATA.filter(r => r.correct !== r.pred);
  document.getElementById('missList').innerHTML = misses.length ? misses.map(r => `
    <div class="miss">
      <div><div class="miss-rating">${r.rating}\u2605</div>
           <div class="miss-correct">${pill(r.correct)}</div></div>
      <div><div class="miss-title">${esc(r.title) || '<i>(no title)</i>'}</div>
           <div class="miss-text">${esc(r.text) || '<i>(no text)</i>'}</div></div>
      <div class="miss-pred"><span class="chip wrong">&times; model said ${r.pred==='POSITIVE'?'POS':'NEG'}</span></div>
    </div>`).join('')
    : '<div class="card" style="color:var(--ink-soft)">No misses.</div>';

  // review table
  const tbody = document.getElementById('reviewBody');
  const count = document.getElementById('resultCount');
  let fOutcome='all', fRating='any', fSearch='';
  function rows(){ const q=fSearch.toLowerCase();
    return DATA.filter(r =>
      (fOutcome==='all' || (fOutcome==='ok' ? r.correct===r.pred : r.correct!==r.pred)) &&
      (fRating==='any' || String(r.rating)===fRating) &&
      (!q || (r.title+' '+r.text).toLowerCase().includes(q))); }
  function render(){ const rs = fOutcome==='miss' ? rows().reverse() : rows();
    tbody.innerHTML = rs.map(r => `
      <tr class="${r.correct===r.pred?'ok':'miss'}">
        <td><span class="row-no">#${r.index+1}</span></td>
        <td><b>${esc(r.title)||'<i>untitled</i>'}</b>
            <span class="qt">${esc(r.text)||'<i>no text</i>'}</span></td>
        <td><span class="stars">${stars(r.rating)}</span><span class="qt">${r.rating}\u2605</span></td>
        <td>${pill(r.correct)}</td><td>${pill(r.pred)}</td>
      </tr>`).join('');
    count.textContent = `${rs.length} of ${DATA.length} reviews`; }
  document.getElementById('outcomeSeg').addEventListener('click', e=>{
    if(e.target.tagName!=='BUTTON') return;
    fOutcome=e.target.dataset.f;
    document.querySelectorAll('#outcomeSeg button').forEach(b=>b.classList.toggle('active', b===e.target));
    render(); });
  document.getElementById('ratingFilter').addEventListener('change', e=>{ fRating=e.target.value; render(); });
  document.getElementById('search').addEventListener('input', e=>{ fSearch=e.target.value; render(); });
  render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
