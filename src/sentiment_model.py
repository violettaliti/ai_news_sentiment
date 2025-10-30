"""
This tiny sentiment_model.py prototype is a sentiment classifier.
- SentimentModel.predict(texts) -> List[str]   # labels: 'pos' | 'neu' | 'neg'
- SentimentModel.predict_proba(texts) -> List[dict]  # {'pos': p, 'neu': p, 'neg': p}
"""
from __future__ import annotations
from typing import Iterable, List, Dict

class SentimentModel1:
    def __init__(self,
                 positive_words: Iterable[str] | None = None,
                 negative_words: Iterable[str] | None = None,
                 neutral_margin: float = 0.1):
        """
        :param positive_words: iterable of positive tokens (lowercase).
        :param negative_words: iterable of negative tokens (lowercase).
        :param neutral_margin: if |pos_score - neg_score| <= margin -> 'neu'
        """
        # tiny starter lexicons (extend as needed)
        default_pos = {
            "good", "great", "excellent", "positive", "benefit",
            "improve", "growth", "success", "win", "strong",
            "innovation", "breakthrough", "efficient", "boost",
            "opportunity", "promising", "advancement", "progress"
        }
        default_neg = {
            "bad", "worse", "worst", "negative", "risk",
            "decline", "loss", "fail", "weak",
            "problem", "issue", "concern", "crisis", "threat",
            "lawsuit", "collapse", "delay", "bug"
        }

        self.pos_words = set(positive_words or default_pos)
        self.neg_words = set(negative_words or default_neg)
        self.neutral_margin = float(neutral_margin)

    # --- public API ---------------------------------------------------------

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment labels for a list of texts.
        Returns labels in {'pos', 'neu', 'neg'}.
        """
        labels: List[str] = []
        for t in texts:
            try:
                label = self._predict_one(t)
                labels.append(label)
            except Exception:
                # Fail-safe: treat problematic entries as neutral
                labels.append("neu")
        return labels

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Return soft scores for each class per text (not true probabilities, but useful).
        Each dict sums to ~1.0.
        """
        outputs: List[Dict[str, float]] = []
        for t in texts:
            try:
                scores = self._score_one(t)
            except Exception:
                scores = {"pos": 0.33, "neu": 0.34, "neg": 0.33}
            outputs.append(scores)
        return outputs

    # --- internal helpers ---------------------------------------------------

    def _predict_one(self, text: str) -> str:
        scores = self._score_one(text)
        # label with max score, but respect neutral margin
        diff = abs(scores["pos"] - scores["neg"])
        if diff <= self.neutral_margin:
            return "neu"
        return "pos" if scores["pos"] > scores["neg"] else "neg"

    def _score_one(self, text: str) -> Dict[str, float]:
        if not isinstance(text, str) or not text.strip():
            return {"pos": 0.33, "neu": 0.34, "neg": 0.33}

        tokens = text.lower().split()

        pos_hits = sum(1 for tok in tokens if tok in self.pos_words)
        neg_hits = sum(1 for tok in tokens if tok in self.neg_words)

        # simple smoothing to avoid divide-by-zero and all-zero vectors
        pos = pos_hits + 1e-6
        neg = neg_hits + 1e-6

        # map to pseudo-probabilities; neutral captures the gap
        total = pos + neg
        pos_score = pos / total
        neg_score = neg / total
        neu_score = max(0.0, 1.0 - (pos_score + neg_score))

        # normalize to ~1
        s = pos_score + neu_score + neg_score
        return {"pos": pos_score / s, "neu": neu_score / s, "neg": neg_score / s}

class SentimentModel2:
    def __init__(
        self,
        positive_words: Iterable[str] | None = None,
        negative_words: Iterable[str] | None = None,
        negations: Iterable[str] | None = None,
        neutral_threshold: float = 0.05,   # score in [-1,1]; |score| <= threshold -> 'neu'
        exclamation_boost: float = 0.05,   # per '!' boost of magnitude
        negative_weight: float = 1.2,      # negatives count a bit stronger than positives
        norm_exponent: float = 0.7,        # sublinear normalization exponent
    ):
        self.neutral_threshold = float(neutral_threshold)
        self.exclamation_boost = float(exclamation_boost)
        self.negative_weight = float(negative_weight)
        self.norm_exponent = float(norm_exponent)

        self.pos_words, self.neg_words = self._load_lexicon(positive_words, negative_words)
        self.negations = set(
            negations
            or {"not", "no", "never", "kein", "keine", "keinen", "keinem", "keiner", "niemals", "nicht"}
        )

    # ----------------------- public API -------------------------------------

    def predict(self, texts: List[str]) -> List[str]:
        """Return labels 'pos' | 'neu' | 'neg' for a list of texts."""
        out: List[str] = []
        for t in texts:
            try:
                score = self._score_one(t)
                out.append(self._label_from_score(score))
            except Exception:
                out.append("neu")
        return out

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Return soft scores per class; they sum ~1.0."""
        probs: List[Dict[str, float]] = []
        for t in texts:
            try:
                score = self._score_one(t)  # in [-1, 1]
            except Exception:
                score = 0.0
            # map continuous score to tri-class “probabilities”
            # center is neutral; convert with simple triangular mapping
            pos = max(0.0,  score)
            neg = max(0.0, -score)
            neu = 1.0 - max(pos, neg)      # neutral occupies the remaining peak mass
            s = pos + neu + neg
            probs.append({"pos": pos / s, "neu": neu / s, "neg": neg / s})
        return probs

    # ----------------------- internals --------------------------------------

    def _label_from_score(self, score: float) -> str:
        if score >  self.neutral_threshold:
            return "pos"
        if score < -self.neutral_threshold:
            return "neg"
        return "neu"

    def _load_lexicon(
        self,
        pos_override: Iterable[str] | None,
        neg_override: Iterable[str] | None
    ) -> Tuple[set[str], set[str]]:
        if pos_override is not None and neg_override is not None:
            return set(map(str.lower, pos_override)), set(map(str.lower, neg_override))
        try:
            import nltk
            from nltk.corpus import opinion_lexicon
            nltk.download("opinion_lexicon", quiet=True)
            print("Loaded NLTK opinion lexicon.")
            return set(opinion_lexicon.positive()), set(opinion_lexicon.negative())
        except Exception:
            print("Could not load NLTK opinion lexicon — using fallback lists.")
            default_pos = {
                "good", "great", "excellent", "positive", "benefit", "improve", "growth",
                "success", "strong", "innovation", "progress", "promising", "win", "record",
                "surge", "boost", "better", "advance", "breakthrough", "efficient"
            }
            default_neg = {
                "bad", "worse", "worst", "negative", "risk", "concern", "decline", "drop",
                "loss", "fail", "weak", "problem", "issue", "crisis", "threat", "fraud",
                "lawsuit", "collapse", "delay", "scandal", "breach"
            }
            return default_pos, default_neg

    def _tokenize(self, text: str) -> List[str]:
        # very light tokenization; trim one trailing sentence punctuation
        toks = []
        for tok in text.lower().split():
            if tok and tok[-1] in ".,;:!?":
                tok = tok[:-1]
            if tok:
                toks.append(tok)
        return toks

    def _score_one(self, text: str) -> float:
        """Return a signed score in [-1, 1]."""
        if not isinstance(text, str) or not text.strip():
            return 0.0

        tokens = self._tokenize(text)

        score_raw = 0.0
        positives = 0
        negatives = 0
        flip_next = False

        for tok in tokens:
            if tok in self.negations:
                flip_next = True
                continue

            val = 0.0
            if tok in self.pos_words:
                val = 1.0
                positives += 1
            elif tok in self.neg_words:
                val = -self.negative_weight   # negatives weigh a bit more
                negatives += 1

            if val != 0.0:
                if flip_next:
                    val = -val
                    flip_next = False
                score_raw += val
            else:
                # no sentiment word; if negation was pending, expire it
                if flip_next:
                    flip_next = False

        # sublinear length normalization (prevents long texts from flattening scores)
        relevant = positives + negatives
        norm = (relevant ** self.norm_exponent) if relevant > 0 else 1.0
        score = score_raw / norm

        # exclamation emphasis
        excls = text.count("!")
        if excls > 0:
            boost = excls * self.exclamation_boost
            score = score + boost if score >= 0 else score - boost

        # clamp to [-1, 1]
        if score > 1.0:
            score = 1.0
        elif score < -1.0:
            score = -1.0
        return score


# --- CLI demo ---------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "AI shows strong progress with promising innovation and great results.",
        "There are concerns about risks and problems in the AI rollout.",
        "Mixed signals today: improvements but also delays and issues reported.",
        "",  # edge case
        None,  # type: ignore
    ]

    print("\n-------------Predicting with model 1 ----------\n")

    model1 = SentimentModel1(neutral_margin=0.12)
    labels1 = model1.predict(samples)
    probs1 = model1.predict_proba(samples)

    print("Predictions:")
    for s, y, p in zip(samples, labels1, probs1):
        print(f"- {y.upper():<3} | {s}\n  -> {p}")

    print("\n\n-------------Predicting with model 2 ----------\n")

    model2 = SentimentModel2(neutral_threshold=0.08, exclamation_boost=0.06)
    labels2 = model2.predict(samples)
    probs2  = model2.predict_proba(samples)

    print("\nPredictions:")
    for s, y, p in zip(samples, labels2, probs2):
        print(f"- {y.upper():<3} | {s}\n  -> {p}")


