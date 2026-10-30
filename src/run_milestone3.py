import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from baseline import build_pipeline, evaluate, predict_with_confidence
from data_prep import (DATA_DIR, PROJECT_ROOT, load_challenge_set, prepare_data,
                       prepare_test, training_pool)

OUT = PROJECT_ROOT / "results" / "m3"
FIG = OUT / "figures"
LABELS = {0: "negative", 1: "positive"}

STRENGTHS = {
    "LogReg": [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
    "NaiveBayes": [0.01, 0.03, 0.1, 0.3, 1.0],
}
FEATURES = {
    "words": dict(ngram_range=(1, 1), negation=False),
    "words+pairs": dict(ngram_range=(1, 2), negation=False),
    "words+neg": dict(ngram_range=(1, 1), negation=True),
    "words+pairs+neg": dict(ngram_range=(1, 2), negation=True),
}
COLORS = {"words": "#9db4d6", "words+pairs": "#2f5f9e", "words+neg": "#f0a868", "words+pairs+neg": "#c0504d"}


def challenge_scores(pipe, negation_set, other_set):
    preds, _ = predict_with_confidence(pipe, negation_set["clean_text"])
    correct = preds == negation_set["label_id"].values
    pairs = pd.Series(correct).groupby(negation_set["pair_id"].values).all()
    other_preds, _ = predict_with_confidence(pipe, other_set["clean_text"])
    other_ok = other_preds == other_set["label_id"].values
    return {
        "neg_sentences": int(correct.sum()),
        "neg_pairs": int(pairs.sum()),
        "mixed": int(other_ok[(other_set["category"] == "mixed").values].sum()),
        "sarcasm": int(other_ok[(other_set["category"] == "sarcasm").values].sum()),
    }


def tune(train_df, val_df):
    rows = []
    for clf, grid in STRENGTHS.items():
        for feat, kw in FEATURES.items():
            for strength in grid:
                pipe = build_pipeline(clf, strength=strength, **kw)
                pipe.fit(train_df["clean_text"], train_df["label"])
                m = evaluate(val_df["label"], pipe.predict(val_df["clean_text"]))
                rows.append({"model": clf, "features": feat, "strength": strength, **m})
    tuning = pd.DataFrame(rows)
    tuning.round(4).to_csv(OUT / "tuning.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, clf in zip(axes, STRENGTHS):
        for feat in FEATURES:
            d = tuning[(tuning.model == clf) & (tuning.features == feat)]
            ax.plot(d["strength"], d["f1"], marker="o", label=feat, color=COLORS[feat])
        ax.set_xscale("log")
        ax.set_xlabel("C (higher = less regularisation)" if clf == "LogReg" else "alpha (higher = more smoothing)")
        ax.set_title("Logistic Regression" if clf == "LogReg" else "Naive Bayes")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Validation F1")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "tuning.png", dpi=150)
    plt.close(fig)

    best = tuning.loc[tuning.groupby(["model", "features"])["f1"].idxmax()]
    return {(r.model, r.features): r.strength for r in best.itertuples()}


def compare(train_df, val_df, best, neg_set, other_set):
    rows = []
    for (clf, feat), strength in best.items():
        pipe = build_pipeline(clf, strength=strength, **FEATURES[feat])
        start = time.perf_counter()
        pipe.fit(train_df["clean_text"], train_df["label"])
        rows.append({
            "model": clf, "features": feat, "strength": strength,
            "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
            "fit_seconds": round(time.perf_counter() - start, 2),
            **evaluate(val_df["label"], pipe.predict(val_df["clean_text"])),
            **challenge_scores(pipe, neg_set, other_set),
        })
    table = pd.DataFrame(rows)
    order = {f: i for i, f in enumerate(FEATURES)}
    table = table.sort_values(["model", "features"], key=lambda c: c.map(order) if c.name == "features" else c)
    table.round(3).to_csv(OUT / "comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(2)
    for i, feat in enumerate(FEATURES):
        d = table[table.features == feat].set_index("model").loc[["LogReg", "NaiveBayes"]]
        axes[0].bar(x + (i - 1.5) * 0.2, d["f1"], 0.2, label=feat, color=COLORS[feat])
        axes[1].bar(x + (i - 1.5) * 0.2, d["neg_pairs"], 0.2, label=feat, color=COLORS[feat])
    for ax, title in zip(axes, ("Validation F1 (800 real reviews)", "Negation pairs fully correct (of 10)")):
        ax.set_xticks(x, ["Logistic Regression", "Naive Bayes"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.8, 0.92)
    axes[1].set_ylim(0, 10)
    axes[1].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "comparison.png", dpi=150)
    plt.close(fig)
    return table


def stability(best, neg_set, other_set, seeds=(1, 2, 3, 4, 5)):
    rows = []
    for seed in seeds:
        train_df, val_df = prepare_data(seed=seed)
        for feat, kw in FEATURES.items():
            pipe = build_pipeline("LogReg", strength=best[("LogReg", feat)], **kw)
            pipe.fit(train_df["clean_text"], train_df["label"])
            rows.append({"seed": seed, "features": feat,
                         **evaluate(val_df["label"], pipe.predict(val_df["clean_text"])),
                         **challenge_scores(pipe, neg_set, other_set)})
    runs = pd.DataFrame(rows)
    runs.round(4).to_csv(OUT / "stability_runs.csv", index=False)
    summary = runs.groupby("features", sort=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), f1_min=("f1", "min"), f1_max=("f1", "max"),
        neg_pairs_mean=("neg_pairs", "mean"), neg_pairs_min=("neg_pairs", "min"),
        neg_pairs_max=("neg_pairs", "max"),
    ).reset_index()
    summary.round(3).to_csv(OUT / "stability_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    x = np.arange(len(summary))
    colors = [COLORS[f] for f in summary["features"]]
    axes[0].bar(x, summary["f1_mean"], yerr=summary["f1_std"], capsize=5, color=colors)
    axes[0].set_ylim(0.83, 0.91)
    axes[0].set_title("Validation F1, mean ± std of 5 splits")
    for i, feat in enumerate(summary["features"]):
        vals = runs.loc[runs.features == feat, "neg_pairs"]
        axes[1].bar(i, vals.mean(), color=colors[i])
        axes[1].scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals,
                        color="black", s=14, zorder=3)
    axes[1].set_ylim(0, 10)
    axes[1].set_title("Negation pairs correct (bar = mean, dots = splits)")
    for ax in axes:
        ax.set_xticks(x, summary["features"], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "stability.png", dpi=150)
    plt.close(fig)
    return summary


def scope(train_df, val_df, best, neg_set, other_set):
    rows = []
    strength = best[("LogReg", "words+pairs+neg")]
    for window in (1, 2, 3, None):
        pipe = build_pipeline("LogReg", (1, 2), negation=True, window=window, strength=strength)
        pipe.fit(train_df["clean_text"], train_df["label"])
        rows.append({
            "window": window or "to punctuation",
            **evaluate(val_df["label"], pipe.predict(val_df["clean_text"])),
            **challenge_scores(pipe, neg_set, other_set),
        })
    table = pd.DataFrame(rows)
    table.round(3).to_csv(OUT / "negation_scope.csv", index=False)
    return table


def learning_curve(val_df, best, neg_set, other_set, sizes=(3200, 6400, 12800, None)):
    rows = []
    for n in sizes:
        pool = training_pool(val_df, n_total=n)
        for feat in ("words+pairs", "words+pairs+neg"):
            pipe = build_pipeline("LogReg", strength=best[("LogReg", feat)], **FEATURES[feat])
            pipe.fit(pool["clean_text"], pool["label"])
            rows.append({
                "train_size": len(pool), "features": feat,
                **evaluate(val_df["label"], pipe.predict(val_df["clean_text"])),
                **challenge_scores(pipe, neg_set, other_set),
            })
            print("  learning curve: {:>6} reviews, {:16s} F1 {:.3f}  pairs {}/10".format(len(pool), feat, rows[-1]['f1'], rows[-1]['neg_pairs']))
    table = pd.DataFrame(rows)
    table.round(3).to_csv(OUT / "learning_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for feat in ("words+pairs", "words+pairs+neg"):
        d = table[table.features == feat]
        axes[0].plot(d["train_size"], d["f1"], marker="o", label=feat, color=COLORS[feat])
        axes[1].plot(d["train_size"], d["neg_pairs"], marker="o", label=feat, color=COLORS[feat])
    axes[0].set_title("Validation F1")
    axes[1].set_title("Negation pairs fully correct (of 10)")
    axes[1].set_ylim(0, 10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(table["train_size"].unique(), [f"{s:,}" for s in table["train_size"].unique()])
        ax.minorticks_off()
        ax.set_xlabel("Training reviews (Logistic Regression)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.png", dpi=150)
    plt.close(fig)
    return table


def choose_final(comparison):
    best_f1 = comparison["f1"].max()
    tied = comparison[comparison["f1"] >= best_f1 - 0.005]
    return tied.sort_values(["neg_pairs", "f1"], ascending=False).iloc[0]


def final_test(val_df, final, neg_set, other_set):
    full_train = pd.concat([training_pool(val_df), val_df], ignore_index=True)
    test = prepare_test()
    models = {
        "M2 baseline (LogReg, words+pairs, 3,200 reviews)":
            (build_pipeline("LogReg", (1, 2)), prepare_data()[0]),
        f"Final ({final.model}, {final.features}, 25,000 reviews)":
            (build_pipeline(final.model, strength=final.strength, **FEATURES[final.features]), full_train),
    }
    rows, fitted = [], {}
    for name, (pipe, train) in models.items():
        pipe.fit(train["clean_text"], train["label"])
        rows.append({"model": name, "train_reviews": len(train),
                     **evaluate(test["label"], pipe.predict(test["clean_text"])),
                     **challenge_scores(pipe, neg_set, other_set)})
        fitted[name] = pipe
    table = pd.DataFrame(rows)
    table.round(3).to_csv(OUT / "test_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_predictions(
            test["label"], pipe.predict(test["clean_text"]), display_labels=["neg", "pos"],
            ax=ax, colorbar=False, cmap="Blues", values_format=",", text_kw={"fontsize": 12})
        ax.set_title(name.split(" (")[0], fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "test_confusion.png", dpi=150)
    plt.close(fig)
    return table, fitted, test


def save_predictions(fitted, neg_set, other_set):
    for name, df in (("negation", neg_set), ("mixed_sarcasm", other_set)):
        out = df.drop(columns=["clean_text", "label_id"]).copy()
        for model_name, pipe in fitted.items():
            short = "baseline" if model_name.startswith("M2") else "final"
            preds, conf = predict_with_confidence(pipe, df["clean_text"])
            out[short] = [f"{LABELS[p]} ({c:.2f})" for p, c in zip(preds, conf)]
            out[f"{short}_ok"] = preds == df["label_id"].values
        out.to_csv(OUT / f"predictions_{name}.csv", index=False)


def save_test_errors(pipe, test, n=10):
    labels, conf = predict_with_confidence(pipe, test["clean_text"])
    wrong = test.assign(pred=labels, conf=conf)[labels != test["label"].values]
    worst = wrong.sort_values("conf", ascending=False).head(n)
    worst = worst.assign(
        true=worst["label"].map(LABELS), pred=worst["pred"].map(LABELS),
        excerpt=worst["text"].str.replace("<br />", " ").str.slice(0, 400),
    )
    worst[["review_id", "true", "pred", "conf", "excerpt"]].round(3).to_csv(OUT / "test_confident_errors.csv", index=False)
    print("  final model: {:,} test errors of {:,}".format(len(wrong), len(test)))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train_df, val_df = prepare_data()
    neg_set = load_challenge_set()
    other_set = load_challenge_set(DATA_DIR / "challenge_mixed_sarcasm.csv")

    print("== 1. Tuning ==")
    best = tune(train_df, val_df)
    for k, v in best.items():
        print("  {:10s} {:16s} best strength {}".format(k[0], k[1], v))

    print("\n== 2. Negation handling (tuned settings, 3,200 training reviews) ==")
    comparison = compare(train_df, val_df, best, neg_set, other_set)
    print(comparison.round(3).to_string(index=False))

    print("\n== 2b. Stability over 5 random splits (LogReg) ==")
    print(stability(best, neg_set, other_set).round(3).to_string(index=False))

    print("\n== 3. Negation scope (LogReg, words+pairs+neg) ==")
    print(scope(train_df, val_df, best, neg_set, other_set).round(3).to_string(index=False))

    print("\n== 4. Learning curve ==")
    learning_curve(val_df, best, neg_set, other_set)

    final = choose_final(comparison)
    print("\n== 5. Final test: chosen model = {}, {}, strength {} ==".format(final.model, final.features, final.strength))
    table, fitted, test = final_test(val_df, final, neg_set, other_set)
    print(table.round(3).to_string(index=False))
    save_predictions(fitted, neg_set, other_set)
    save_test_errors(fitted[[k for k in fitted if k.startswith("Final")][0]], test)
    print("\nSaved to {}".format(OUT))


if __name__ == "__main__":
    main()
