import time
from functools import partial

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from negation import mark_negation

RANDOM_SEED = 42

CLASSIFIERS = {
    "LogReg": lambda strength=1.0: LogisticRegression(C=strength, max_iter=2000, random_state=RANDOM_SEED),
    "NaiveBayes": lambda strength=1.0: MultinomialNB(alpha=strength),
}

NGRAMS = {
    "A: words (1,1)": (1, 1),
    "B: words+pairs (1,2)": (1, 2),
}


def build_pipeline(classifier="LogReg", ngram_range=(1, 1), negation=False, window=None, strength=1.0):
    vectorizer = TfidfVectorizer(
        preprocessor=partial(mark_negation, window=window) if negation else None,
        ngram_range=ngram_range,
        stop_words=None,
        min_df=2,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", CLASSIFIERS[classifier](strength))])


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def run_experiments(train_df, val_df, text_col="clean_text"):
    results, models = [], {}

    dummy = DummyClassifier(strategy="most_frequent").fit(train_df[text_col], train_df["label"])
    results.append({
        "model": "Majority class", "version": "-", "n_features": 0, "fit_seconds": 0.0,
        **evaluate(val_df["label"], dummy.predict(val_df[text_col])),
    })

    for clf_name in CLASSIFIERS:
        for version, ngram_range in NGRAMS.items():
            pipe = build_pipeline(clf_name, ngram_range)
            start = time.perf_counter()
            pipe.fit(train_df[text_col], train_df["label"])
            fit_seconds = time.perf_counter() - start
            preds = pipe.predict(val_df[text_col])
            results.append({
                "model": clf_name,
                "version": version,
                "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
                "fit_seconds": round(fit_seconds, 2),
                **evaluate(val_df["label"], preds),
            })
            models[(clf_name, version)] = pipe
    return results, models


def predict_with_confidence(pipe, texts):
    proba = pipe.predict_proba(texts)
    labels = proba.argmax(axis=1)
    return labels, proba.max(axis=1)


def top_features(pipe, n=15):
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    order = coefs.argsort()
    negative = [(vocab[i], round(coefs[i], 3)) for i in order[:n]]
    positive = [(vocab[i], round(coefs[i], 3)) for i in order[::-1][:n]]
    return positive, negative
