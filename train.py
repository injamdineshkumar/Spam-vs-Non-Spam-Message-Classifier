
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_classifier.joblib"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "spam.csv not found. Prepare the dataset first."
        )

    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Support common dataset column formats.
    columns = {str(c).lower().strip(): c for c in df.columns}

    if "label" in columns and "message" in columns:
        label_col = columns["label"]
        message_col = columns["message"]
    elif "v1" in columns and "v2" in columns:
        label_col = columns["v1"]
        message_col = columns["v2"]
    else:
        raise ValueError(
            "CSV must contain label/message or v1/v2 columns."
        )

    df = df[[label_col, message_col]].copy()
    df.columns = ["label", "message"]
    df = df.dropna()

    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
    )
    df["message"] = df["message"].astype(str).str.strip()

    df["label"] = df["label"].replace({
        "non-spam": "ham",
        "nonspam": "ham"
    })

    df = df[
        df["label"].isin(["ham", "spam"])
        & (df["message"] != "")
    ]

    if df["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both ham and spam messages."
        )

    return df


def main():
    df = load_dataset()

    print("Total messages:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # Fit TF-IDF on training messages only.
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    k = min(3000, X_train_tfidf.shape[1])

    # Keep preprocessing, feature selection and classifier together.
    model = Pipeline([
        ("tfidf", tfidf),
        ("chi2", SelectKBest(score_func=chi2, k=k)),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:",
          round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion matrix (ham, spam):")
    print(confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ))

    joblib.dump(model, MODEL_PATH)
    print("\nTrained model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
