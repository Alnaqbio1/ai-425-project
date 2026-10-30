import sys

from baseline import NGRAMS, build_pipeline, predict_with_confidence
from data_prep import clean_text, prepare_data

LABELS = {0: "negative", 1: "positive"}


def train_models():
    train_df, _ = prepare_data()
    return {
        version: build_pipeline("LogReg", ngram).fit(train_df["clean_text"], train_df["label"])
        for version, ngram in NGRAMS.items()
    }


def show(models, review):
    text = clean_text(review)
    print('\nReview: "{}"'.format(review))
    for version, pipe in models.items():
        (label,), (conf,) = predict_with_confidence(pipe, [text])
        print("  LogReg {:22s} -> {:8s} (confidence {:.2f})".format(version, LABELS[label], conf))


def main():
    print("Training baseline models on 3,200 IMDb reviews ...")
    models = train_models()
    if len(sys.argv) > 1:
        for review in sys.argv[1:]:
            show(models, review)
        return
    while True:
        review = input("\nEnter a movie review (empty line to quit): ").strip()
        if not review:
            break
        show(models, review)


if __name__ == "__main__":
    main()
