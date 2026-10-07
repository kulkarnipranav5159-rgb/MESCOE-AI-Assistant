"""
MESCOE chatbot - training script (intent classification)

Pipeline
  1. Load + clean data (missing values, duplicates, text normalisation)
  2. Stratified 80/20 train/test split
  3. Features: TF-IDF on words (1-2 grams) + TF-IDF on characters (2-5 grams).
     The character n-grams make the model robust to typos ("fess", "hostle").
  4. Four models, each tuned with 5-fold GridSearchCV on the TRAIN set only:
       Logistic Regression, Linear SVM (calibrated), Multinomial Naive Bayes, Random Forest
  5. Evaluate on (a) the held-out test split and (b) a hand-written CHALLENGE set of
     questions worded differently from anything in the training data (incl. typos).
  6. Best model = highest cross-validated F1 (ties broken by challenge accuracy).
  7. Refit best model on all data and save everything the Streamlit app needs.

Run:  python train_model.py
"""
import json
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from nlp_utils import clean_text

RANDOM_STATE = 42

# ---------------------------------------------------------------- 1. data
df = pd.read_csv("mescoe_dataset.csv")
print("Missing values per column:\n", df.isnull().sum().to_string(), "\n")
df = df.dropna(subset=["text", "intent", "response"])
df["clean"] = df["text"].map(clean_text)
n_before = len(df)
df = df.drop_duplicates(subset="clean").reset_index(drop=True)
print(f"Rows: {n_before} -> {len(df)} after removing duplicates | intents: {df['intent'].nunique()}")

# ---------------------------------------------------------------- challenge set
# Unseen questions (different wording, some with typos) used as an honest stress test.
CHALLENGE = [
    ("hey good evening", "greeting"), ("hello is this the mescoe bot", "greeting"),
    ("ok i will leave now", "goodbye"), ("see ya", "goodbye"),
    ("thanks that solved my doubt", "thanks"), ("thank you very much for the help", "thanks"),
    ("where exactly is wadia college in pune", "location"),
    ("how do i reach there from the station", "location"),
    ("wher is the colege", "location"),
    ("which streams can i take in engineering", "courses"),
    ("what degrees are on offer", "courses"),
    ("what are the steps to get admission", "admission_process"),
    ("do i need mht cet to join", "admission_process"),
    ("give me the number of the admission office", "admission_contact"),
    ("email id for admission questions", "admission_contact"),
    ("how many students does the computer department take", "intake_computer"),
    ("seats in comp engg", "intake_computer"),
    ("what is the seat count for e&tc department", "intake_entc"),
    ("how many students are taken in electronics and telecom", "intake_entc"),
    ("mechanical department seat capacity", "intake_mech"),
    ("how many seats does mech have", "intake_mech"),
    ("number of seats in the robotics department", "intake_robotics"),
    ("automation and robotics seat count", "intake_robotics"),
    ("how much will i have to pay each year", "fees"),
    ("fess for engeneering", "fees"),
    ("can outstation students stay in a hostel", "hostel"),
    ("hostle facilty", "hostel"),
    ("which companies hire students from here", "placements"),
    ("placment record", "placements"),
    ("does the college have a naac grade", "accreditation"),
    ("is the college linked with pune university", "accreditation"),
    ("does the library have e books", "library"),
    ("how many books are there to read", "library"),
    ("is there any financial help for poor students", "scholarships"),
    ("what is the tfws scheme", "scholarships"),
    ("till what time does the office stay open", "office_timings"),
    ("is the admin office open on saturdays", "office_timings"),
    ("can i get lunch in the college", "canteen"),
    ("is there a cafeteria on campus", "canteen"),
    ("what outdoor games are there", "sports"),
    ("does the college have a football ground", "sports"),
    ("which person is the principal here", "principal"),
    ("who is the head of the institute", "principal"),
    ("can i do masters here", "postgraduate"),
    ("which me branches are available", "postgraduate"),
    ("what papers do i need to bring", "documents_required"),
    ("do i have to submit a domicile certificate", "documents_required"),
    ("tell me about the annual fest", "events"),
    ("are there any student chapters like ieee", "events"),
    ("where can i download my hall ticket", "exams"),
    ("when will the exam timetable come out", "exams"),
    ("what are you capable of", "bot_info"),
    ("are you a real person", "bot_info"),
    ("what is the weather like in mumbai", "out_of_scope"),
    ("tell me something funny", "out_of_scope"),
    ("who is the president of the usa", "out_of_scope"),
    ("asdfgh", "out_of_scope"), ("xyz qwerty", "out_of_scope"),
    ("kkkk", "out_of_scope"), ("12345", "out_of_scope"),
]
chal = pd.DataFrame(CHALLENGE, columns=["text", "intent"])
chal["clean"] = chal["text"].map(clean_text)
overlap = chal["clean"].isin(set(df["clean"]))
if overlap.any():
    print(f"Dropping {overlap.sum()} challenge questions that also appear in training data")
    chal = chal[~overlap].reset_index(drop=True)
print(f"Challenge set: {len(chal)} unseen questions\n")

# ---------------------------------------------------------------- 2. split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["intent"], test_size=0.2, stratify=df["intent"], random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# ---------------------------------------------------------------- 3. features + models
def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                 strip_accents="unicode")),
    ])


def make_pipe(clf):
    return Pipeline([("features", make_features()), ("clf", clf)])


candidates = {
    "Logistic Regression": (
        make_pipe(LogisticRegression(max_iter=3000)),
        {"clf__C": [1, 10, 50, 100]},
    ),
    "Linear SVM (calibrated)": (
        make_pipe(CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
        {"clf__estimator__C": [1, 3, 10]},
    ),
    "Multinomial Naive Bayes": (
        make_pipe(MultinomialNB()),
        {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
    ),
    "Random Forest": (
        make_pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        {},
    ),
}

# ---------------------------------------------------------------- 4/5. tune + evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, fitted, test_preds = [], {}, {}

for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    preds = best.predict(X_test)
    fitted[name], test_preds[name] = best, preds
    results.append({
        "Model": name,
        "CV F1 (train)": round(gs.best_score_, 4),
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
        "Challenge Accuracy": round(accuracy_score(chal["intent"], best.predict(chal["clean"])), 4),
        "Best Params": json.dumps(gs.best_params_) if gs.best_params_ else "-",
    })
    print(f"  done: {name}")

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("model_metrics.csv", index=False)
print("\n" + metrics_df.drop(columns="Best Params").to_string(index=False))

# ---------------------------------------------------------------- 6. choose best
ranked = metrics_df.sort_values(["CV F1 (train)", "Challenge Accuracy"], ascending=False)
best_name = ranked.iloc[0]["Model"]
print(f"\nBest model: {best_name}")

with open("classification_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n=== Held-out test set ===\n")
    f.write(classification_report(y_test, test_preds[best_name], zero_division=0))
    f.write("\n=== Challenge set (unseen wording) ===\n")
    f.write(classification_report(chal["intent"], fitted[best_name].predict(chal["clean"]),
                                  zero_division=0))

fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay.from_predictions(
    y_test, test_preds[best_name], xticks_rotation=90, ax=ax, colorbar=False
)
ax.set_title(f"Confusion Matrix - {best_name} (held-out test set)")
plt.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150)

# ---------------------------------------------------------------- confidence threshold
# Pick the cut-off that gives the best accuracy on the challenge set when low-confidence
# predictions are treated as "out_of_scope". The app lets the user adjust it.
probs = fitted[best_name].predict_proba(chal["clean"])
classes = fitted[best_name].classes_
top_idx, top_conf = probs.argmax(axis=1), probs.max(axis=1)
best_t, best_acc = 0.30, -1
for t in [x / 100 for x in range(20, 71, 5)]:
    pred = [classes[i] if c >= t else "out_of_scope" for i, c in zip(top_idx, top_conf)]
    acc = accuracy_score(chal["intent"], pred)
    if acc > best_acc:
        best_t, best_acc = t, acc
print(f"Recommended confidence threshold: {best_t:.2f} (challenge accuracy {best_acc:.3f})")

# ---------------------------------------------------------------- 7. refit on all data + save
final_model = clone(fitted[best_name]).fit(df["clean"], df["intent"])
joblib.dump(final_model, "mescoe_chatbot_pipeline.pkl")
with open("model_config.json", "w") as f:
    json.dump({"best_model": best_name, "confidence_threshold": best_t,
               "n_samples": int(len(df)), "n_intents": int(df["intent"].nunique())}, f, indent=2)
print("Saved: mescoe_chatbot_pipeline.pkl, model_metrics.csv, model_config.json, "
      "confusion_matrix.png, classification_report.txt")
