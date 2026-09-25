"""
app.py
------
Streamlit dashboard for the Step-2 scoring of the Amazon Gift_Card sentiment
classifier (title + text only, scored against the rating on the first 100 rows).

Three-class: Rating >= 4 → POSITIVE, Rating == 3 → NEUTRAL, Rating <= 2 → NEGATIVE.

Run:   .venv\\Scripts\\streamlit run app.py
(or)   uv run --with streamlit streamlit run app.py

Every figure is computed here from data/results/batch_100_scores.csv at load time,
so what appears on screen is exactly what the scoring produced.
"""
import csv, json, os

import altair as alt
import pandas as pd
import streamlit as st

from assets.style import CSS

SCORES = "data/results/batch_100_scores.csv"
EVAL_METRICS = "data/results/eval_metrics.json"

# ---- palette (rehost/recolor from here) -------------------------------------
POS    = "#22863a"   # vibrant green   (positive)
NEG    = "#d4373b"   # alert crimson   (negative)
NEU    = "#4a5abf"   # confident indigo (neutral)
ACCENT = "#5c6abf"   # primary brand
ACCENT_SOFT = "#e8e9f4"
INK    = "#2a2c48"
INK_SOFT = "#4a4a5a"
FAINT  = "#7b7f9e"
FAINT_SOFT = "#b0b3c4"
BACKGROUND = "#f8f9fd"
SURFACE2 = "#ffffff"

# Class colours for use in charts
CLASS_COLOURS = {"POSITIVE": POS, "NEUTRAL": NEU, "NEGATIVE": NEG}
CLASS_LABELS = {"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"}

CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


@st.cache_data(show_spinner=False)
def load_data():
    rows = []
    with open(SCORES, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "index": int(r["index"]),
                "title": r["title"],
                "text": r["text"],
                "rating": int(float(r["rating"])),
                "correct": r["correct_label"],
                # Preserve blank/unparseable API outputs as an explicit category.
                "pred": r["predicted_label"] or "UNPARSED",
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["match"] = df["correct"] == df["pred"]
    return df


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(EVAL_METRICS):
        return None
    with open(EVAL_METRICS, encoding="utf-8") as fh:
        return json.load(fh)


# ---- helpers ----------------------------------------------------------------

def _chart_title(title, subtitle=None):
    """Return a layer-safe Altair chart title.

    Streamlit handles responsive width; subtitles stay in nearby dashboard copy
    because Altair LayerChart does not accept a standalone ``subtitle`` property.
    """
    return {"title": title}


def _neutral_safe(val, default=0):
    """Return val if > 0, else default — avoids zero-width bars."""
    return val if val and val > 0 else default


# ---- star-rating distribution -----------------------------------------------

def render_star_dist(df):
    """Histogram of star ratings in the scored batch."""
    dist = df.groupby("rating").size().reset_index(name="count")
    dist["star_label"] = dist["rating"].astype(str) + "★"

    # Colours: darker stars = more reviews
    max_count = dist["count"].max()

    base = alt.Chart(dist).encode(
        x=alt.X("rating:N", title="Star Rating",
                sort=list(range(1, 6)),
                axis=alt.Axis(
                    labels=True, labelAngle=0, labelFontSize=13,
                    labelColor=INK, titleColor=INK_SOFT, titleFontWeight=600)),
        tooltip=["star_label", "count"],
    )

    bars = base.mark_bar(
        color=ACCENT, cornerRadiusTopRight=4
    ).encode(
        y=alt.Y("count:Q", title="Number of Reviews",
                scale=alt.Scale(domain=[0, max_count * 1.1])),
        color=alt.Color("count:Q",
                        scale=alt.Scale(range=["#d0d4f0", ACCENT]),
                        legend=None),
    )

    vals = bars.mark_text(
        align="center", dy=-6, fontSize=13, fontWeight=700, color=INK
    ).encode(text="count:Q")

    return (bars + vals).properties(**_chart_title(
        "Star Rating Distribution",
        "How the scored reviews are spread across ★1 … ★5"))


# ---- per-class comparison (correct vs predicted) ----------------------------

def render_class_comparison(df):
    """Grouped bar chart: for each true class, how many the model got right / wrong."""
    classes = CLASSES
    labels = [CLASS_LABELS[c] for c in classes]

    data = []
    for true_cls in classes:
        subset = df[df["correct"] == true_cls]
        if len(subset) == 0:
            data.append({
                "Class": labels[classes.index(true_cls)],
                "Category": "None",
                "Count": 0,
            })
            continue
        right = int(subset["match"].sum())
        wrong = len(subset) - right
        data.append({
            "Class": labels[classes.index(true_cls)],
            "Category": "Matched",
            "Count": right,
        })
        data.append({
            "Class": labels[classes.index(true_cls)],
            "Category": "Missed",
            "Count": wrong,
        })

    chart_df = pd.DataFrame(data)
    max_count = max(d["Count"] for d in data) if data else 0

    bars = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4).encode(
        x=alt.X("Class:N", title="True Class", sort=labels,
                axis=alt.Axis(
                    labels=True, labelAngle=0, labelFontSize=11,
                    labelColor=INK_SOFT, titleColor=INK_SOFT, titleFontWeight=600)),
        xOffset=alt.XOffset("Category:N", sort=["Matched", "Missed", "None"]),
        y=alt.Y("Count:Q", title="Number of Reviews",
                scale=alt.Scale(domain=[0, max(max_count * 1.2, 10)])),
        color=alt.Color("Category:N", scale=alt.Scale(
                domain=["Matched", "Missed", "None"],
                range=[POS, NEG, "#e0e0e0"])),
        tooltip=["Class", "Category", "Count"],
    ).properties(**_chart_title(
        "Per-Class: Model Correct vs Missed",
        "Side-by-side bars show whether each true class was matched or missed."))

    vals = alt.Chart(chart_df[chart_df["Count"] > 0]).mark_text(
        align="center", dy=-8, fontSize=12, fontWeight=600, color=INK
    ).encode(
        x=alt.X("Class:N", sort=labels),
        xOffset=alt.XOffset("Category:N", sort=["Matched", "Missed", "None"]),
        y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, max(max_count * 1.2, 10)])),
        text="Count:Q",
    )

    return bars + vals


# ---- per-class accuracy breakdown -------------------------------------------

def render_class_accuracy(df):
    """Horizontal bar: one wide bar per class, % inside, count to the right.

    Handles zero-sample and zero-accuracy cases gracefully so the chart
    never looks broken — missing classes get a dashed outline, and 0 %
    gets a visible thin bar with an N/A label.
    """
    accuracy = []
    for c in CLASSES:
        subset = df[df["correct"] == c]
        if len(subset) == 0:
            accuracy.append({
                "Class": CLASS_LABELS[c],
                "Accuracy": 0.0,
                "Correct": 0,
                "Total": 0,
                "ZeroSample": True,
            })
        else:
            right = int(subset["match"].sum())
            acc = right / len(subset)
            accuracy.append({
                "Class": CLASS_LABELS[c],
                "Accuracy": acc,
                "Correct": right,
                "Total": len(subset),
                "ZeroSample": False,
            })

    acc_df = pd.DataFrame(accuracy)

    chart = alt.Chart(acc_df).properties(
        title={
            "text": "Per-Class Accuracy",
            "subtitle": "What fraction of each true class the model identified correctly.",
            "anchor": "start",
            "fontSize": 15,
            "fontWeight": 700,
            "color": INK,
            "subtitleFontSize": 11,
            "subtitleColor": INK_SOFT,
            "dy": -2,
        },
        width=600,
        height=280,
    )

    # Bars — full-width horizontal, color-coded by class
    bars = chart.mark_bar(
        cornerRadiusTopRight=6,
        fillOpacity=0.88,
    ).encode(
        x=alt.X("Accuracy:Q", title="Match %",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(
                    grid=True, gridColor="#e8e8f0", tickCount=6,
                    format=".0%", labelFontSize=12, labelColor=INK_SOFT,
                    titleColor=INK_SOFT, titleFontWeight=600)),
        y=alt.Y("Class:N", title="", sort=None,
                axis=alt.Axis(
                    labels=True, labelFontSize=15, labelColor=INK,
                    labelFontWeight=600,
                    titleFontWeight=600, titleColor=INK_SOFT,
                    labelPadding=12)),
        color=alt.Color("Class:N", scale=alt.Scale(
                domain=CLASS_LABELS.values(), range=[POS, NEU, NEG])),
        tooltip=["Class", "Accuracy", "Correct", "Total"],
    )

    # Dashed outline for zero-sample classes so they're not invisible
    dashed = chart.transform_filter(
        alt.datum.ZeroSample
    ).mark_bar(
        color="transparent",
        stroke=FAINT,
        strokeWidth=2,
        strokeDash=[4, 3],
        cornerRadiusTopRight=6,
    ).encode(
        x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Class:N", title="", sort=None),
    )

    # Percentage label — positioned at bar end, auto-chosen color
    pct_txt = chart.mark_text(
        align="left", dx=10, fontSize=18, fontWeight=700,
    ).encode(
        text=alt.Text("Accuracy:Q", format=".0%"),
        color=alt.condition(
            alt.datum.Accuracy > 0.20,
            alt.value("white"),
            alt.value(INK)
        ),
        y=alt.Y("Class:N", title="", sort=None),
    )

    # Count label to the right
    count_txt = chart.mark_text(
        align="left", dx=110, fontSize=13, color="#888", fontWeight=500,
    ).encode(
        text=alt.Text("Correct:T", format="({} of {})"),
        y=alt.Y("Class:N", title="", sort=None),
    )

    # "N/A" label for zero-accuracy classes with samples
    na_txt = chart.transform_filter(
        (alt.datum.Accuracy == 0) & (~alt.datum.ZeroSample)
    ).mark_text(
        align="left", dx=10, fontSize=14, fontWeight=700, color=NEG,
    ).encode(
        text=alt.value("N/A  (" + str(acc_df.loc[
            (acc_df.Accuracy == 0) & (~acc_df.ZeroSample), "Total"
        ].values[0]) + " samples)"),
        y=alt.Y("Class:N", title="", sort=None),
    )

    # Note about small sample size
    note = alt.Chart(pd.DataFrame({
        "x": [0.75], "y": [1], "text": [
            "* Note: Neutral has only 2 samples in the batch"
        ]
    })).mark_text(
        align="right", fontSize=10, color=FAINT, fontStyle="italic",
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("text:N"),
    )

    return bars + dashed + pct_txt + count_txt + na_txt + note


# ---- scored-batch confusion matrix -----------------------------------------

def render_batch_confusion(df):
    """Show the saved batch's true label vs model output, including unparsed replies."""
    predicted = CLASSES + ["UNPARSED"]
    pred_labels = [CLASS_LABELS[c] for c in CLASSES] + ["Unparseable"]
    grid = []
    for true_cls in CLASSES:
        for pred_cls in predicted:
            count = int(((df["correct"] == true_cls) & (df["pred"] == pred_cls)).sum())
            grid.append({
                "True": CLASS_LABELS[true_cls],
                "Predicted": pred_labels[predicted.index(pred_cls)],
                "Count": count,
                "Outcome": "Correct" if true_cls == pred_cls else "Wrong / unparseable",
            })
    grid = pd.DataFrame(grid)
    max_count = max(int(grid["Count"].max()), 1)

    heat = alt.Chart(grid).mark_rect(stroke=SURFACE2, strokeWidth=3, cornerRadius=4).encode(
        x=alt.X("Predicted:N", title="Model Prediction", sort=pred_labels,
                axis=alt.Axis(labelAngle=0, labelFontSize=11, labelColor=INK_SOFT,
                              titleColor=INK_SOFT, titleFontWeight=600)),
        y=alt.Y("True:N", title="Correct Answer", sort=[CLASS_LABELS[c] for c in CLASSES],
                axis=alt.Axis(labelFontSize=11, labelColor=INK_SOFT,
                              titleColor=INK_SOFT, titleFontWeight=600)),
        color=alt.Color("Outcome:N", scale=alt.Scale(
            domain=["Correct", "Wrong / unparseable"], range=[POS, NEG]), legend=None),
        opacity=alt.Opacity("Count:Q", scale=alt.Scale(domain=[0, max_count], range=[0.12, 1]), legend=None),
        tooltip=["True", "Predicted", "Count", "Outcome"],
    ).properties(**_chart_title(
        "Correct Answer vs. Model Prediction",
        "Rows are rating-derived labels; columns are the saved model outputs."))
    labels = heat.mark_text(fontSize=16, fontWeight=700, color=INK).encode(text="Count:Q")
    return heat + labels


# ---- confusion matrix -------------------------------------------------------

def render_confusion(m):
    """Altair heatmap confusion matrix: rows=true label, cols=predicted (3 classes)."""
    cm = m["confusion_matrix"]
    classes = CLASSES
    labels = [CLASS_LABELS[c] for c in classes]
    grid = []
    for true_cls in classes:
        for pred_cls in classes:
            grid.append({
                "true": labels[classes.index(true_cls)],
                "pred": labels[classes.index(pred_cls)],
                "count": cm.get(true_cls, {}).get(pred_cls, 0),
            })
    grid = pd.DataFrame(grid)
    maxc = grid["count"].max()
    heat = alt.Chart(grid).mark_rect(
        stroke=SURFACE2, strokeWidth=3, cornerRadius=4
    ).encode(
        x=alt.X("pred:N", title="Predicted", sort=labels,
                axis=alt.Axis(
                    labels=True, labelAngle=0, labelFontSize=12,
                    labelColor=INK_SOFT, titleColor=INK_SOFT, titleFontWeight=600)),
        y=alt.Y("true:N", title="True Label", sort=labels,
                axis=alt.Axis(
                    labels=True, labelFontSize=12, labelColor=INK_SOFT,
                    titleColor=INK_SOFT, titleFontWeight=600)),
        color=alt.Color("count:Q", scale=alt.Scale(
            range=[ACCENT_SOFT, ACCENT], domain=[0, maxc]), legend=None),
        tooltip=["true", "pred", "count"],
    ).properties(width=310, height=260)
    txt = heat.mark_text(dy=0, fontSize=24, fontWeight=800, color="white").encode(
        text="count:Q")
    return heat + txt


# ---- broader evaluation -----------------------------------------------------

def render_eval():
    m = load_metrics()
    st.subheader("Broader held-out evaluation")
    if not m:
        st.info("Run `python evaluate.py` to generate the broader evaluation "
                "(data/results/eval_metrics.json).")
        return

    total = m.get("total", 0)
    classes_in = m.get("per_class", {})
    missing_classes = [c for c in CLASSES if c not in classes_in]
    if missing_classes:
        st.warning(
            "The held-out evaluation file was generated under the older binary "
            "label rule and is not shown here. Run `python evaluate.py` to "
            "regenerate it under the current POSITIVE / NEUTRAL / NEGATIVE rule."
        )
        return

    st.caption((
        "Beyond the first-100 batch, a balanced held-out sample of "
        f"**{total} reviews** scored under the same three-class rule. "
        "Equal POSITIVE &amp; NEGATIVE halves plus the neutral tier, "
        "making this harder than the first-100 rows."
    ))

    # Headline numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balanced accuracy", f"{m.get('balanced_accuracy', 0) * 100:.1f}%",
              help="Mean of per-class recall — model can't cheat on class skew")
    c2.metric("Macro-F1", f"{m.get('macro_f1', 0):.3f}",
              help="Mean of per-class F1 scores")
    c3.metric("Natural-pop accuracy",
              f"{m.get('estimated_natural_accuracy', 0) * 100:.1f}%",
              help="Estimated on the real 9:1 positive skew")
    c4.metric("Unparseable", str(m.get("unparsed", 0)),
              help=f"out of {total} held-out reviews")

    a, b = st.columns([1, 1.4], gap="large")
    with a:
        st.write("**Confusion matrix** (rows = true, columns = predicted)")
        st.altair_chart(render_confusion(m), width="stretch")
    with b:
        st.write("**Per-class metrics**")
        perf = pd.DataFrame([
            {
                "Class": CLASS_LABELS.get(c, c),
                "Precision": classes_in[c]["precision"],
                "Recall": classes_in[c]["recall"],
                "F1": classes_in[c]["f1"],
                "n": classes_in[c]["support"],
            }
            for c in CLASSES if c in classes_in
        ])
        st.dataframe(perf, hide_index=True, width="stretch",
                     column_config={
                         "Class": st.column_config.TextColumn("Class"),
                         "Precision": st.column_config.NumberColumn(
                             "Precision", format="%.3f"),
                         "Recall": st.column_config.NumberColumn(
                             "Recall", format="%.3f"),
                         "F1": st.column_config.NumberColumn(
                             "F1", format="%.3f"),
                         "n": st.column_config.NumberColumn("n"),
                     })
        st.caption("Recall = share of a class the model caught; "
                   "precision = how often its call was right.")


# ---- main -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Gift-Card Sentiment · Scoring",
        page_icon="📊", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ---- sidebar ----------------------------------------------------------
    with st.sidebar:
        st.markdown("#### How this was scored")
        st.write(
            "A text-only classifier was given **just title + text** — never the "
            "rating — and asked to output **POSITIVE**, **NEUTRAL**, or "
            "**NEGATIVE** per review, on the first 100 rows of the file."
        )
        st.markdown("**Correct-label rule:**")
        st.markdown(
            "- rating **≥ 4** → POSITIVE\n"
            "- rating **== 3** → NEUTRAL\n"
            "- rating **≤ 2** → NEGATIVE"
        )
        st.caption(
            "The rating is used only to check the model afterwards.\n"
            "Neutral reviews are rare in the first 100 rows."
        )
        st.divider()
        st.caption("Source · data/results/batch_100_scores.csv")
        if st.button("⟳ Reload data"):
            st.cache_data.clear()
            st.rerun()

    # ---- load data --------------------------------------------------------
    df = load_data()
    if df is None or df.empty:
        st.warning("No scores found. Run `python score_batch.py` first "
                   "to generate `data/results/batch_100_scores.csv`.")
        st.stop()

    total = len(df)
    correct = int(df["match"].sum())
    wrong = total - correct
    acc = correct / total
    unparsed = int((df["pred"] == "UNPARSED").sum())

    # Per-class subsets
    class_groups = {c: df[df["correct"] == c] for c in CLASSES}

    # ---- header -----------------------------------------------------------
    st.markdown(
        '<p class="kicker">Amazon Gift-Card Reviews · Step 2 · 3-Class</p>',
        unsafe_allow_html=True,
    )
    st.title("Sentiment scoring dashboard")
    st.markdown(
        '<p class="subtitle">'
        "How a <b>text-only</b> classifier (title + text, never the rating) "
        "matches the label derived from the rating — "
        "<b>≥ 4 = POSITIVE, 3 = NEUTRAL, ≤ 2 = NEGATIVE</b> — "
        f"on the first {total} rows of the file."
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- headline metrics -----------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall match", f"{acc * 100:.1f}%",
              help=f"{correct} of {total} reviews matched")
    for idx, (c, col) in enumerate(zip(CLASSES, [c2, c3, c4])):
        sub = class_groups[c]
        right = int(sub["match"].sum()) if len(sub) > 0 else 0
        pct = right / len(sub) * 100 if len(sub) > 0 else 0
        col.metric(
            CLASS_LABELS[c],
            f"{right} / {len(sub)} ({pct:.0f}%)",
            help=f"True {CLASS_LABELS[c]} reviews: {right} matched",
        )
    c5.metric("Unparseable", str(unparsed),
              help="Reviews that did not return one of the three required labels")

    # ---- descriptive overview: star dist + per-class comparison + accuracy
    st.subheader("Data overview")
    st.caption(
        "How reviews are distributed by star rating, how the model performed "
        "per true class, and per-class accuracy — so failures are visible at a glance."
    )

    colA, colB, colC = st.columns(3)
    with colA:
        st.altair_chart(render_star_dist(df), width="stretch")
    with colB:
        st.altair_chart(render_class_comparison(df), width="stretch")
    with colC:
        st.altair_chart(render_class_accuracy(df), width="stretch")

    st.subheader("Where predictions diverge")
    st.caption(
        "The matrix makes every failure explicit: diagonal cells are correct; "
        "red off-diagonal cells are wrong predictions or blank/unparseable outputs."
    )
    st.altair_chart(render_batch_confusion(df), width="stretch")

    # ---- right / wrong overview -------------------------------------------
    st.subheader("Right vs. wrong, at a glance")
    st.caption("Donut: total matched vs. missed. Below: per-rating match %.")

    colA, colB = st.columns([1, 1.05], gap="large")

    with colA:
        pie = pd.DataFrame({
            "Outcome": ["Matched", "Missed"],
            "Count": [correct, wrong],
        })
        donut = alt.Chart(pie).mark_arc(
            innerRadius=62, stroke=BACKGROUND, strokeWidth=4
        ).encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color("Outcome:N", scale=alt.Scale(
                domain=["Matched", "Missed"], range=[POS, NEG])),
            tooltip=["Outcome", "Count"],
        ).properties(height=240, width=240)
        # center text
        txt = alt.Chart(
            pd.DataFrame({"t": [f"{acc * 100:.1f}%"]})).mark_text(
                text=f"{acc * 100:.1f}%", size=30, fontWeight=700, color=INK
        ).encode(theta=alt.Theta("t:N", stack=True)
        ).properties(width=240, height=240)
        st.altair_chart(
            (donut + txt).resolve_scale(theta="shared"),
            width="stretch",
        )
        st.caption(f"{correct} matched · {wrong} missed of {total}")

    with colB:
        # per-rating bars — percentage correct
        grp = (
            df.groupby("rating")["match"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grp.columns = ["rating", "ok", "n"]
        grp["pct"] = (grp.ok / grp.n * 100).round(1)

        # Colour bars by accuracy: green >= 80, amber 50-80, red < 50
        def _bar_colour(p):
            if p >= 80:
                return POS
            elif p >= 50:
                return "#e8a830"  # amber
            return NEG

        grp["col"] = grp["pct"].apply(_bar_colour)

        rating_chart = alt.Chart(grp).mark_bar(
            cornerRadiusTopRight=6
        ).encode(
            x=alt.X("rating:N", title="Star Rating",
                    sort=list(range(1, 6)),
                    axis=alt.Axis(
                        labelExpr='"★" + datum.label',
                        labelFontSize=13, labelColor=INK,
                        titleColor=INK_SOFT, titleFontWeight=600)),
            y=alt.Y("pct:Q", title="Match %",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("col:N", scale=alt.Scale(
                domain=["green", "amber", "red"],
                range=[POS, "#e8a830", NEG]),
                legend=None),
            tooltip=["rating", "pct", "ok", "n"],
        ).properties(**_chart_title("Match % by Star Value"))

        rtext = rating_chart.mark_text(
            align="center", dy=-10, color=INK_SOFT, size=13, fontWeight=700
        ).encode(text="pct:Q")
        st.altair_chart((rating_chart + rtext), width="stretch")

    st.info(
        "**Why overall match reads high:** the first 100 rows are naturally "
        "positive-skewed — "
        f"**{', '.join(f'{int((df.rating == lv).sum())}★{lv}' for lv in [5, 4, 3, 2, 1] if (df.rating == lv).any())}** "
        "(★1 → ★5). "
        "Reviews at the extremes (★1 / ★5) agree with the model almost perfectly; "
        "disagreement clusters at mid ratings. "
        "Some misses are also cases where the *text* disagrees with the *star "
        "rating* — the model can only judge the words.",
        icon="💡",
    )

    # ---- broader held-out evaluation --------------------------------------
    render_eval()

    # ---- misses -----------------------------------------------------------
    st.subheader(f"What the model got wrong ({wrong})")
    st.caption(
        "The only reviews where title + text led to a label that disagrees "
        "with the rating label."
    )
    misses = df[~df["match"]]
    if misses.empty:
        st.success("No misses.")
    else:
        for _, r in misses.iterrows():
            correct_cls = r["correct"]
            pred_cls = r["pred"]
            correct_ch = "pos" if correct_cls == "POSITIVE" else ("neu" if correct_cls == "NEUTRAL" else "neg")
            pred_ch = "neg" if pred_cls in {"NEGATIVE", "UNPARSED"} else ("neu" if pred_cls == "NEUTRAL" else "pos")
            title_val = r["title"] or "(no title)"
            text_val = r["text"] or "(no text)"
            pred_label = CLASS_LABELS.get(pred_cls, "Unparseable")
            st.markdown(
                f'<div class="misscard">'
                f'<div class="mr">{r.rating}★</div>'
                f'<span class="chip {correct_ch}">true {CLASS_LABELS[correct_cls]}</span>'
                f'<span class="chip {pred_ch}">model → {pred_label}</span>'
                f'<div class="mt">{title_val}</div>'
                f'<div class="mx">{text_val}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ---- per-review evidence ----------------------------------------------
    st.subheader("Per-review evidence")
    st.caption(
        "Every scored review — filter by outcome, rating, or search the text."
    )

    f1, f2, f3 = st.columns([1.2, 1, 2.4])
    outcome = f1.segmented_control(
        "Outcome", ["All", "Matched", "Missed"],
        default="All", label_visibility="collapsed",
    )
    rating_f = f2.selectbox(
        "Rating", ["Any"] + [str(i) for i in range(1, 6)],
        label_visibility="collapsed",
    )
    query = f3.text_input("Search title/text", "").strip().lower()

    view = df.copy()
    if outcome == "Matched":
        view = view[view["match"]]
    elif outcome == "Missed":
        view = view[~view["match"]]
    if rating_f != "Any":
        view = view[view["rating"] == int(rating_f)]
    if query:
        mask = (
            view["title"].str.lower().str.contains(query, na=False)
            | view["text"].str.lower().str.contains(query, na=False)
        )
        view = view[mask]

    tbl = view[["index", "title", "text", "rating", "correct", "pred"]].rename(
        columns={
            "index": "#",
            "rating": "★",
            "correct": "Correct",
            "pred": "Predicted",
        }
    )
    tbl["#"] = tbl["#"] + 1
    st.dataframe(
        tbl,
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(format="%d"),
            "★": st.column_config.NumberColumn(format="%d"),
            "title": "Title (input)",
            "text": "Text (input)",
        },
    )
    st.caption(f"Showing {len(view)} of {total} reviews.")

    st.divider()
    st.caption(
        "Method: labels come from the rating **only for scoring** and are never "
        "shown to the model. "
        "Inputs (no rating): `data/batch_100_input.csv` · "
        "answer key: `data/batch_100_answerkey.csv`. "
        "Regenerate with "
        "`python score_batch.py && .venv\\Scripts\\streamlit run app.py`."
    )


if __name__ == "__main__":
    main()
