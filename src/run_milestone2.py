from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from baseline import predict_with_confidence, run_experiments, top_features
from data_prep import PROJECT_ROOT, load_challenge_set, prepare_data

RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
LABELS = {0: "negative", 1: "positive"}


def save_data_stats(train_df, val_df):
    rows = []
    for name, df in (("train", train_df), ("validation", val_df)):
        words = df["clean_text"].str.split().str.len()
        rows.append({
            "split": name, "reviews": len(df),
            "positive": int((df["label"] == 1).sum()),
            "negative": int((df["label"] == 0).sum()),
            "mean_words": round(words.mean(), 1),
            "median_words": int(words.median()),
            "max_words": int(words.max()),
            "reviews_with_not": f"{df['clean_text'].str.contains(r'\bnot\b').mean():.0%}",
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS_DIR / "data_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = train_df["label"].map(LABELS).value_counts().reindex(["negative", "positive"])
    axes[0].bar(counts.index, counts.values, color=["#c0504d", "#4f81bd"])
    axes[0].set_title("Class balance (training split)")
    axes[0].set_ylabel("Reviews")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center")
    axes[0].set_ylim(0, counts.max() * 1.15)

    for label, color in ((0, "#c0504d"), (1, "#4f81bd")):
        lengths = train_df.loc[train_df["label"] == label, "clean_text"].str.split().str.len()
        axes[1].hist(lengths.clip(upper=1000), bins=40, alpha=0.6, color=color, label=LABELS[label])
    axes[1].set_title("Review length (words, capped at 1000)")
    axes[1].set_xlabel("Words per review")
    axes[1].set_ylabel("Reviews")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_overview.png", dpi=150)
    plt.close(fig)
    return stats


def save_results(results):
    table = pd.DataFrame(results)
    for col in ("accuracy", "precision", "recall", "f1"):
        table[col] = table[col].round(3)
    table.to_csv(RESULTS_DIR / "validation_results.csv", index=False)
    (RESULTS_DIR / "validation_results.md").write_text(table.to_markdown(index=False))
    return table


def save_confusion_matrices(models, val_df):
    fig, axes = plt.subplots(1, len(models), figsize=(11, 3.1))
    for ax, ((clf, version), pipe) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_predictions(
            val_df["label"], pipe.predict(val_df["clean_text"]),
            display_labels=["neg", "pos"], ax=ax, colorbar=False, cmap="Blues",
            text_kw={"fontsize": 13},
        )
        ax.set_title(f"{clf}\n{version}", fontsize=11)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def save_challenge_results(models, challenge):
    out = challenge[["pair_id", "variant", "text", "label"]].copy()
    summary = []
    for (clf, version), pipe in models.items():
        short = f"{clf} {version.split(':')[0]}"
        preds, conf = predict_with_confidence(pipe, challenge["clean_text"])
        out[short] = [f"{LABELS[p]} ({c:.2f})" for p, c in zip(preds, conf)]
        correct = preds == challenge["label_id"].values
        out[f"{short} ok"] = correct
        pair_ok = pd.Series(correct).groupby(challenge["pair_id"].values).all()
        summary.append({
            "model": clf, "version": version,
            "sentences_correct": f"{correct.sum()}/{len(correct)}",
            "base_correct": f"{correct[challenge['variant'] == 'base'].sum()}/10",
            "negated_correct": f"{correct[challenge['variant'] == 'negated'].sum()}/10",
            "pairs_fully_correct": f"{pair_ok.sum()}/10",
        })
    out.to_csv(RESULTS_DIR / "challenge_predictions.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS_DIR / "challenge_summary.csv", index=False)
    return out, summary


def save_top_features(models):
    lines = []
    for version in ("A: words (1,1)", "B: words+pairs (1,2)"):
        pos, neg = top_features(models[("LogReg", version)], n=15)
        lines.append(f"## LogReg {version}\n")
        lines.append("| most positive | weight | most negative | weight |\n|---|---|---|---|")
        lines += [f"| {p[0]} | {p[1]} | {n[0]} | {n[1]} |" for p, n in zip(pos, neg)]
        lines.append("")
    (RESULTS_DIR / "top_features.md").write_text("\n".join(lines))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df = prepare_data()

    print("== Data ==")
    print(save_data_stats(train_df, val_df).to_string(index=False))

    results, models = run_experiments(train_df, val_df)
    print("\n== Validation results ==")
    print(save_results(results).to_string(index=False))

    save_confusion_matrices(models, val_df)
    save_top_features(models)

    challenge = load_challenge_set()
    preds, summary = save_challenge_results(models, challenge)
    print("\n== Negation challenge set ==")
    print(summary.to_string(index=False))
    print("\nSaved tables and figures to {}".format(RESULTS_DIR))


if __name__ == "__main__":
    main()
