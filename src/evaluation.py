# evaluation.py
import pandas as pd
from src.sentiment_model import SentimentModel1, SentimentModel2 # make sure this file is in the same folder

class Evaluator:
    """
    Simple evaluator for SentimentModel outputs.

    Works with a list of dicts returned by SentimentModel.predict_proba(),
    e.g. [{"pos":0.7,"neu":0.2,"neg":0.1}, ...]
    """

    def __init__(self, results):
        """
        :param results: list of dicts with keys 'pos', 'neu', 'neg'
        """
        if not results or not isinstance(results, list):
            raise ValueError("Results must be a non-empty list of dictionaries.")
        self.df = pd.DataFrame(results)
        if not {"pos", "neu", "neg"}.issubset(self.df.columns):
            raise ValueError("Each result must contain 'pos', 'neu', and 'neg' keys.")

    # basic summaries
    def average_scores(self):
        """Return average positive, neutral, and negative scores."""
        return self.df.mean().to_dict()

    def overall_distribution(self):
        """
        Return distribution of predicted labels.
        Predicted label = argmax(pos, neu, neg)
        """
        labels = self.df.idxmax(axis=1)
        counts = labels.value_counts()
        total = len(labels)
        shares = (counts / total).round(3)
        return pd.DataFrame({
            "count": counts,
            "share": shares
        }).reset_index().rename(columns={"index": "sentiment"})

    def summary(self):
        """Combined overview with mean scores and label distribution."""
        avg = self.average_scores()
        dist = self.overall_distribution()
        return {
            "mean_scores": avg,
            "distribution": dist
        }

    def top_positive(self, texts, n=5):
        """Return the top n texts with highest positive score."""
        df = self.df.copy()
        df["text"] = texts
        return df.sort_values("pos", ascending=False).head(n)[["pos", "neu", "neg", "text"]]

    def top_negative(self, texts, n=5):
        """Return the top n texts with highest negative score."""
        df = self.df.copy()
        df["text"] = texts
        return df.sort_values("neg", ascending=False).head(n)[["pos", "neu", "neg", "text"]]

if __name__ == "__main__":
    # Example texts
    texts = [
        "AI brings amazing progress and innovation!",
        "There are serious concerns about data privacy risks.",
        "Mixed opinions about how AI will affect jobs.",
        "AI development seems neutral overall.",
    ]

    # get probabilities from SentimentModel1
    model = SentimentModel1()
    results = model.predict_proba(texts)

    # evaluate them
    evaluator = Evaluator(results)

    print("\n--- Evaluating sentiment model 1 results ---")

    print("\n--- Average scores ---")
    print(evaluator.average_scores())

    print("\n--- Label distribution ---")
    print(evaluator.overall_distribution())

    print("\n--- Top positive texts ---")
    print(evaluator.top_positive(texts))

    print("\n--- Top negative texts ---")
    print(evaluator.top_negative(texts))

    print("\n--- Moving to the next model ---")

    # get probabilities from SentimentModel2
    model = SentimentModel2()
    results = model.predict_proba(texts)

    # evaluate them
    evaluator = Evaluator(results)

    print("\n--- Evaluating sentiment model 2 results ---")

    print("\n--- Average scores ---")
    print(evaluator.average_scores())

    print("\n--- Label distribution ---")
    print(evaluator.overall_distribution())

    print("\n--- Top positive texts ---")
    print(evaluator.top_positive(texts))

    print("\n--- Top negative texts ---")
    print(evaluator.top_negative(texts))
