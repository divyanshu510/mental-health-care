import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional lemmatization (falls back gracefully if nltk data isn't available)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _lemmatizer = WordNetLemmatizer()
    def _lemmatize(word):
        return _lemmatizer.lemmatize(word)
except Exception:
    def _lemmatize(word):
        return word

# ---------------------------------------------------------------
# 1) LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("neww.csv")

df.rename(columns={"statement": "text", "status": "label"}, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(subset=["text"], inplace=True)

print("Original dataset size:", df.shape)
print(df["label"].value_counts())

# ---------------------------------------------------------------
# 2) AUGMENT WITH SHORT "I am / I feel" TEMPLATE PHRASES
#
# The raw dataset almost only contains short "I am X" phrases under
# "Normal" and almost only contains short "I feel X" phrases under the
# other classes. That means the model was learning "am" = Normal and
# "feel" = not-Normal, instead of learning what the emotion word itself
# means. Adding a small, balanced set of "I am ___" / "I feel ___"
# examples for every class breaks that spurious correlation.
# ---------------------------------------------------------------
emotion_words = {
    "Normal": ["happy", "glad", "great", "good", "fine", "calm", "content",
               "relaxed", "cheerful", "peaceful"],
    "Depression": ["sad", "depressed", "hopeless", "empty", "worthless",
                   "unmotivated", "numb", "low", "miserable", "down"],
    "Anxiety": ["anxious", "nervous", "scared", "worried", "panicked",
                "uneasy", "restless", "tense", "fearful", "on edge"],
    "Stress": ["stressed", "overwhelmed", "exhausted", "burned out",
               "overworked", "pressured", "drained", "frazzled",
               "tense", "worn out"],
    "Bipolar": ["manic", "euphoric", "on top of the world",
                "extremely energetic", "emotionally all over the place",
                "up and down"],
    "Personality Disorder": ["empty inside", "afraid of being abandoned",
                              "unsure who I am", "unstable",
                              "disconnected from myself"],
}

templates = ["I am {}", "I feel {}", "I am feeling {}",
             "I feel so {}", "I am really {}"]

augmented_rows = [
    {"text": t.format(word), "label": label}
    for label, words in emotion_words.items()
    for word in words
    for t in templates
]
aug_df = pd.DataFrame(augmented_rows)

df = pd.concat([df, aug_df], ignore_index=True)
df.drop_duplicates(subset=["text"], inplace=True)

print("\nDataset size after augmentation:", df.shape)
print(df["label"].value_counts())

# ---------------------------------------------------------------
# 3) PREPROCESSING
# ---------------------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [_lemmatize(w) for w in words]
    return " ".join(words)

df["text"] = df["text"].apply(preprocess)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 4) MODEL
#
# Switched from RandomForestClassifier -> LogisticRegression.
# With ~1,100 short text rows and thousands of sparse TF-IDF
# features, RandomForest doesn't have enough samples per split to
# find real patterns, so it ends up close to guessing (all classes
# came out ~15-24% probability for every input). A linear model
# (LogisticRegression / LinearSVC) is the standard, much stronger
# choice for small/medium TF-IDF text classification problems.
# ---------------------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=5,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 5) EVALUATION
# ---------------------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred, labels=model.classes_))

# ---------------------------------------------------------------
# 6) SANITY CHECK ON THE EXACT CASES THAT WERE BROKEN
# ---------------------------------------------------------------
sanity_tests = [
    "I am happy", "I feel happy",
    "I am sad", "I feel sad",
    "I am stressed", "I feel anxious",
    "I feel like crying", "I don't enjoy anything anymore",
]

print("\nSanity checks:")
for t in sanity_tests:
    pred = model.predict([preprocess(t)])[0]
    conf = model.predict_proba([preprocess(t)]).max()
    print(f"  {t!r:40s} -> {pred:22s} ({conf*100:.1f}%)")

# ---------------------------------------------------------------
# 7) SAVE MODEL
# ---------------------------------------------------------------
joblib.dump(model, "mental_health_model.pkl")
print("\nSaved mental_health_model.pkl")