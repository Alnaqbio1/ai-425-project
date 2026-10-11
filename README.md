# Sentiment Analysis of Text Reviews

CSC 425 AI Project, Milestone 1

## Team

| Member | Responsibilities |
|---|---|
| Obaid Alnaqbi | Solo project, all roles: problem definition, data collection & preprocessing, model development, evaluation, documentation, and presentation |

## Project Description

This project builds a system that classifies text reviews (movie/product reviews) as **positive** or **negative** sentiment.

- **Input:** A short piece of free-form text (e.g., a movie or product review).
- **AI Task:** Supervised binary text classification.
- **Output:** A predicted sentiment label (`positive` / `negative`), optionally with a confidence score.

I picked this topic because it's something I run into all the time as a regular user (checking reviews before watching a movie or buying something), and because companies use systems like this to track how people feel about their products without reading every review by hand. It's also a reasonable first ML project for me since there's a lot of existing labeled data for it, so I don't have to build my own dataset from scratch, and since the labels are already known I can actually check whether my model is working instead of just guessing.

## Dataset

- **Source:** [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011), from the Stanford AI Lab.
- **Size:** 50,000 labeled reviews (25,000 train / 25,000 test), balanced positive/negative.
- **Preprocessing:** Lowercasing, HTML/punctuation removal, stopword removal, tokenization, TF-IDF or Bag-of-Words vectorization.

## Proposed Approach

1. **Baseline:** TF-IDF + Logistic Regression / Naive Bayes (scikit-learn).
2. **Stretch goal:** Simple neural network (embedding + feed-forward/LSTM) or fine-tuned DistilBERT, compared against the baseline.
3. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix on held-out test set.

## Repository Structure

```
csc425-sentiment-analysis/
├── data/           # raw and processed datasets (not committed if large; see .gitignore)
├── notebooks/       # exploratory data analysis and prototyping notebooks
├── src/              # reusable Python modules (preprocessing, training, evaluation)
├── reports/         # written milestone reports / proposal
└── README.md
```

## References

1. Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
2. Stanford AI Lab: [IMDB Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
3. Scikit-learn: [Working With Text Data tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
4. Hugging Face: [`datasets`](https://huggingface.co/docs/datasets) and [`transformers`](https://huggingface.co/docs/transformers) documentation
5. Kaggle: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial)

## Timeline

| Period | Target |
|---|---|
| Now – Oct. 11 | Define problem, select dataset, set up GitHub repo |
| Oct. 11 – Oct. 25 | Explore/preprocess data, build baseline classifier |
| Oct. 26 – Nov. 15 | Improve approach: tune baseline, implement stretch-goal model |
| Nov. 16 – Nov. 29 | Evaluate, analyze, and refine |
| Nov. 30 – Dec. 11 | Finalize report, demo, and presentation |
