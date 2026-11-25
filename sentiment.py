"""Shared model code: the same cleaning, augmentation and calibrated SVM as SVM.ipynb."""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).parent
DATA_PATH = HERE / "DailyDialog.csv"
MODEL_PATH = HERE / "sentiment_model.joblib"

SENTIMENT_MAP = {
    "joy": "positive",
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "neutral": "neutral",
}

# Hand-written examples from the notebook that help the minority classes.
EXTRA_DATA = [
    ("I absolutely love this new restaurant", "positive"),
    ("The service here is just amazing", "positive"),
    ("This is the best food I have ever had", "positive"),
    ("What a beautiful and sunny day", "positive"),
    ("I had a terrible experience at that place", "negative"),
    ("The movie was a complete waste of time", "negative"),
    ("I am very disappointed with the product", "negative"),
    ("My computer is on the desk in the office", "neutral"),
    ("The car is parked in the garage", "neutral"),
    ("She is reading a book in the library", "neutral"),
    ("The meeting is scheduled for 3 PM", "neutral"),
    ("He is wearing a blue shirt today", "neutral"),
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["text", "sentiment"])
    df["label"] = df["sentiment"].map(SENTIMENT_MAP)
    df = df.dropna(subset=["label"])[["text", "label"]]
    extra = pd.DataFrame(EXTRA_DATA, columns=["text", "label"])
    return pd.concat([df, extra], ignore_index=True)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                                  max_df=0.8, min_df=5)),
        ("svm", CalibratedClassifierCV(
            LinearSVC(random_state=42, max_iter=2000, dual=True), cv=3)),
    ])


# The training data is ~60% negative, which pulls everyday sentences toward
# "negative". Dividing the probabilities by (class share ** PRIOR_STRENGTH)
# partly undoes that: test accuracy drops ~1 point, but positive recall rises
# from 61% to 69% and neutral recall from 71% to 77%.
PRIOR_STRENGTH = 0.5


def adjusted_proba(bundle: dict, texts) -> np.ndarray:
    probs = bundle["pipeline"].predict_proba(texts) / bundle["prior"] ** PRIOR_STRENGTH
    return probs / probs.sum(axis=1, keepdims=True)


def train(save: bool = True, verbose: bool = True) -> dict:
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    pipeline = build_pipeline().fit(X_train, y_train)
    classes = pipeline.classes_
    prior = y_train.value_counts(normalize=True).reindex(classes).to_numpy()
    bundle = {"pipeline": pipeline, "classes": classes, "prior": prior}
    if verbose:
        y_pred = classes[adjusted_proba(bundle, X_test).argmax(axis=1)]
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
    if save:
        joblib.dump(bundle, MODEL_PATH)
        if verbose:
            print(f"Saved model to {MODEL_PATH}")
    return bundle


def load_model() -> dict:
    """Load the saved model, or train one from the CSV if it is missing."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                return bundle
        except Exception:
            pass  # saved with a different scikit-learn version: retrain below
    return train(save=True, verbose=False)


def key_words(bundle: dict, text: str, label: str, top: int = 4) -> list[str]:
    """Words in `text` that pushed the model most toward `label`.

    Each calibration fold holds its own LinearSVC, so their weights are averaged.
    A word's push is its TF-IDF value times that class's weight.
    """
    pipeline = bundle["pipeline"]
    tfidf = pipeline.named_steps["tfidf"]
    svms = [c.estimator for c in pipeline.named_steps["svm"].calibrated_classifiers_]
    coef = np.mean([s.coef_ for s in svms], axis=0)
    k = list(bundle["classes"]).index(label)
    row = tfidf.transform([text])
    vocab = tfidf.get_feature_names_out()
    pushes = [(vocab[j], v * coef[k, j]) for j, v in zip(row.indices, row.data)]
    return [w for w, p in sorted(pushes, key=lambda x: -x[1]) if p > 0][:top]


def predict(bundle: dict, text: str) -> tuple[str, dict[str, float]]:
    probs = adjusted_proba(bundle, [text])[0]
    scores = dict(zip(bundle["classes"], map(float, probs)))
    return max(scores, key=scores.get), scores
