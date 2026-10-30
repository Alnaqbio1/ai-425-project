import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from baseline import build_pipeline, evaluate
from data_prep import (DATA_DIR, PROJECT_ROOT, load_amazon_reviews, load_challenge_set,
                       load_rotten_tomatoes, prepare_data, prepare_test, training_pool)
from negation import NEGATORS, expand_contractions

OUT = PROJECT_ROOT / "results" / "m4"
FIG = OUT / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
C_GRID = [0.3, 1, 3, 10, 30, 100, 1000]


def expected_calibration_error(y, proba, bins=10):
    conf = np.maximum(proba, 1 - proba)
    correct = (proba >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def scores(pipe, df):
    proba = pipe.predict_proba(df["clean_text"])[:, 1]
    y = df["label"].values
    return {
        **evaluate(y, (proba >= 0.5).astype(int)),
        "log_loss": log_loss(y, proba),
        "brier": brier_score_loss(y, proba),
        "ece": expected_calibration_error(y, proba),
    }


def challenge(pipe, neg_set, other_set):
    p = pipe.predict(neg_set["clean_text"])
    ok = p == neg_set["label_id"].values
    q = pipe.predict(other_set["clean_text"]) == other_set["label_id"].values
    return {
        "neg_pairs": int(pd.Series(ok).groupby(neg_set["pair_id"].values).all().sum()),
        "mixed": int(q[(other_set["category"] == "mixed").values].sum()),
        "sarcasm": int(q[(other_set["category"] == "sarcasm").values].sum()),
    }


def calibrate(val_df):
    pool = training_pool(val_df)
    rows, probas = [], {}
    for C in C_GRID:
        pipe = build_pipeline("LogReg", (1, 2), negation=True, strength=C)
        pipe.fit(pool["clean_text"], pool["label"])
        rows.append({"C": C, **scores(pipe, val_df)})
        probas[C] = pipe.predict_proba(val_df["clean_text"])[:, 1]
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "calibration.csv", index=False)
    best_C = table.loc[table["log_loss"].idxmin(), "C"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(table["C"], table["f1"], marker="o", label="F1", color="#2f5f9e")
    axes[0].plot(table["C"], table["ece"], marker="o", label="calibration error (ECE)", color="#c0504d")
    axes[0].axvline(best_C, color="gray", ls="--", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_title("Validation F1 vs calibration error")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    y = val_df["label"].values
    for C, color in ((best_C, "#2f5f9e"), (1000, "#c0504d")):
        conf = np.maximum(probas[C], 1 - probas[C])
        correct = (probas[C] >= 0.5).astype(int) == y
        edges = np.linspace(0.5, 1.0, 6)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, 4)
        xs = [conf[idx == b].mean() for b in range(5) if (idx == b).any()]
        ys = [correct[idx == b].mean() for b in range(5) if (idx == b).any()]
        axes[1].plot(xs, ys, marker="o", color=color, label=f"C = {C:g}")
    axes[1].plot([0.5, 1], [0.5, 1], ls="--", color="gray", lw=1, label="perfectly honest")
    axes[1].set_xlabel("Confidence the model reports")
    axes[1].set_ylabel("How often it is right")
    axes[1].set_title("Reliability (validation)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)
    return best_C, table


def has_negation(text):
    tokens = set(expand_contractions(text).split())
    return bool(tokens & NEGATORS)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train_df, val_df = prepare_data()
    neg_set = load_challenge_set()
    other_set = load_challenge_set(DATA_DIR / "challenge_mixed_sarcasm.csv")

    print("== 1. Calibration ==")
    best_C, cal = calibrate(val_df)
    print(cal.round(4).to_string(index=False))
    print("  chosen C = {:g} (lowest validation log loss)".format(best_C))

    print("\n== 2. Final system on the test set ==")
    full_train = pd.concat([training_pool(val_df), val_df], ignore_index=True)
    models = {
        "M2 baseline": (build_pipeline("LogReg", (1, 2)), train_df),
        "M3 model": (build_pipeline("LogReg", (1, 2), strength=1000), full_train),
        "Final system": (build_pipeline("LogReg", (1, 2), negation=True, strength=best_C), full_train),
    }
    for name, (pipe, data) in models.items():
        pipe.fit(data["clean_text"], data["label"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(models["Final system"][0], MODEL_PATH)

    datasets = {
        "IMDb test (25,000 reviews)": prepare_test(),
        "Rotten Tomatoes sentences (1,066)": load_rotten_tomatoes(),
        "Amazon product reviews (4,000)": load_amazon_reviews(),
    }
    rows = []
    for dname, df in datasets.items():
        for name, (pipe, data) in models.items():
            rows.append({"dataset": dname, "model": name, "train_reviews": len(data), **scores(pipe, df)})
    results = pd.DataFrame(rows)
    results.round(4).to_csv(OUT / "final_results.csv", index=False)
    print(results.round(3).to_string(index=False))

    print("\n== 3. Sentences with vs without negation (Rotten Tomatoes) ==")
    rt = datasets["Rotten Tomatoes sentences (1,066)"]
    rt["negated"] = rt["clean_text"].map(has_negation)
    rows = []
    for subset, df in (("with negation", rt[rt.negated]), ("without negation", rt[~rt.negated])):
        for name, (pipe, _) in models.items():
            m = evaluate(df["label"], pipe.predict(df["clean_text"]))
            rows.append({"subset": subset, "sentences": len(df), "model": name, "accuracy": m["accuracy"]})
    neg_split = pd.DataFrame(rows)
    neg_split.round(3).to_csv(OUT / "rotten_negation_split.csv", index=False)
    print(neg_split.round(3).to_string(index=False))

    print("\n== 4. Challenge sets ==")
    ch = pd.DataFrame([{"model": n, **challenge(p, neg_set, other_set)} for n, (p, _) in models.items()])
    ch.to_csv(OUT / "challenge_results.csv", index=False)
    print(ch.to_string(index=False))

    final = models["Final system"][0]
    proba = final.predict_proba(neg_set["clean_text"])[:, 1]
    out = neg_set[["pair_id", "variant", "text", "label"]].assign(
        final=[f"{'positive' if p >= 0.5 else 'negative'} ({max(p, 1 - p):.2f})" for p in proba])
    out.to_csv(OUT / "final_negation_predictions.csv", index=False)

    vocab = final.named_steps["tfidf"].get_feature_names_out()
    coef = final.named_steps["clf"].coef_[0]
    neg_feats = pd.DataFrame({"feature": vocab, "weight": coef})
    neg_feats = neg_feats[neg_feats.feature.str.startswith("not_") & ~neg_feats.feature.str.contains(" ")]
    top = pd.concat([neg_feats.nlargest(12, "weight"), neg_feats.nsmallest(12, "weight")])
    top.round(3).to_csv(OUT / "top_not_features.csv", index=False)
    print("\nStrongest not_ features:\n", top.round(2).to_string(index=False))
    print("\nSaved final model to {} and results to {}".format(MODEL_PATH, OUT))


if __name__ == "__main__":
    main()
