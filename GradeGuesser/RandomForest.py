import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, train as xgb_train

# ---------- 1. Load and prepare data ----------
PATH = r"C:\Users\vahurpaist\Downloads\problems_2023_01_30\problems MoonBoard 2016 .json"
with open(PATH, encoding="utf-8") as f:
    payload = json.load(f)

texts, grades = [], []
for item in payload["data"]:
    labels = [m["description"] for m in item["moves"]]
    texts.append(" ".join(labels))
    grades.append(item["grade"])

# ---------- 2. Vectorize and encode ----------
vec = TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2))
X = vec.fit_transform(texts)

le = LabelEncoder()
y = le.fit_transform(grades)  # ordinal target

# ---------- 3. Train/test split for stricter scoring ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 4. Prepare DMatrix for xgb.train ----------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

dtrain = DMatrix(X_tr, label=y_tr)
dval = DMatrix(X_val, label=y_val)

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "max_depth": 12,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "seed": 42,
}

evals = [(dval, "eval"), (dtrain, "train")]

bst = xgb_train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=False,
)

# ---------- 5. Evaluate with graded closeness ----------
# Prepare test data
dtest = DMatrix(X_test)
y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration))

# Round and clip predictions
y_pred_rounded = np.round(y_pred).astype(int)
y_pred_clipped = np.clip(y_pred_rounded, 0, len(le.classes_) - 1)

base = 0.65  # penalty base (stricter < 1)
grade_diff = np.abs(y_pred_clipped - y_test)
credits = base ** grade_diff
graded_score = np.mean(credits)

print(f"\nGraded closeness score (base={base}): {graded_score:.3f}")

diff = np.abs(y_pred_clipped - y_test)
strict_hits = (diff <= 1).sum()
strict_score = strict_hits / len(y_test)

print(f"\nStricter closeness score (±1 grade): {strict_score:.3f}")

# ---------- 6. Retrain on full data ----------
dall = DMatrix(X, label=y)
bst_full = xgb_train(
    params,
    dall,
    num_boost_round=bst.best_iteration or 2000,
    verbose_eval=False,
)

joblib.dump(bst_full, "grade_model.joblib")
joblib.dump(vec, "vectorizer.joblib")
joblib.dump(le, "label_encoder.joblib")

# ---------- 7. Prediction function ----------
def predict_grade(move_labels):
    bst = joblib.load("grade_model.joblib")  # Load the booster model
    vec = joblib.load("vectorizer.joblib")
    le = joblib.load("label_encoder.joblib")

    input_text = " ".join(move_labels)
    input_vector = vec.transform([input_text])
    dinput = DMatrix(input_vector)  # convert to DMatrix for xgboost

    pred_float = bst.predict(dinput)[0]  # use bst and DMatrix
    pred_index = int(round(pred_float))
    pred_index = np.clip(pred_index, 0, len(le.classes_) - 1)
    return le.inverse_transform([pred_index])[0]
