import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train as xgb_train

DATA_PATH = Path(
    r"C:\Users\vahurpaist\Downloads\problems_2023_01_30\problems MoonBoard 2016 .json"
)

THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "grade_model.joblib"
VEC_PATH = THIS_DIR / "vectorizer.joblib"
GRADE_MAP_PATH = THIS_DIR / "grade_map.joblib"

COLUMNS = list("ABCDEFGHIJK")
N_EXTRA_FEATURES = 16
GRADE_ORDER = [
    "6B",
    "6B+",
    "6C",
    "6C+",
    "7A",
    "7A+",
    "7B",
    "7B+",
    "7C",
    "7C+",
    "8A",
    "8A+",
    "8B",
    "8B+",
]
GRADE_TO_IDX = {g: i for i, g in enumerate(GRADE_ORDER)}
IDX_TO_GRADE = {i: g for i, g in enumerate(GRADE_ORDER)}


def hold_to_xy(hold_id: str) -> Tuple[int, int]:
    if not hold_id or len(hold_id) < 2:
        return 0, 0
    col_char = hold_id[0].upper()
    row_part = hold_id[1:]
    try:
        x = COLUMNS.index(col_char)
    except ValueError:
        x = 0
    try:
        y = int(row_part)
    except ValueError:
        y = 0
    return x, y


def choose_label(problem: Dict[str, Any]) -> str:
    grade = problem.get("grade")
    user_grade = problem.get("userGrade") or grade
    if problem.get("upgraded"):
        return user_grade
    if problem.get("downgraded"):
        return user_grade
    return grade


def build_move_tokens_from_item(item: Dict[str, Any]) -> List[str]:
    moves = item.get("moves") or []
    tokens: List[str] = []
    for m in moves:
        h = m.get("description") or ""
        if not h:
            continue
        if m.get("isStart"):
            tokens.append(f"START_{h}")
        elif m.get("isEnd"):
            tokens.append(f"END_{h}")
        else:
            tokens.append(h)
    return tokens


def problem_features(item: Dict[str, Any]) -> np.ndarray:
    moves = item.get("moves") or []
    coords = [hold_to_xy(m.get("description") or "") for m in moves]
    if coords:
        xs = np.array([x for x, _ in coords], dtype=float)
        ys = np.array([y for _, y in coords], dtype=float)
    else:
        xs = np.array([0.0])
        ys = np.array([0.0])
    num_moves = float(len(coords))
    num_starts = float(sum(bool(m.get("isStart")) for m in moves))
    num_ends = float(sum(bool(m.get("isEnd")) for m in moves))
    max_y = float(ys.max())
    min_y = float(ys.min())
    vert_span = max_y - min_y
    horiz_span = float(xs.max() - xs.min())
    if len(coords) > 1:
        deltas = np.diff(np.c_[xs, ys], axis=0)
        dists = np.linalg.norm(deltas, axis=1)
    else:
        dists = np.array([0.0])
    max_dist = float(dists.max())
    mean_dist = float(dists.mean())
    big_moves = float((dists >= 4.0).sum())
    user_rating = float(item.get("userRating") or 0.0)
    repeats = float(item.get("repeats") or 0.0)
    repeats_log = float(np.log1p(repeats))
    is_benchmark = 1.0 if item.get("isBenchmark") else 0.0
    method = (item.get("method") or "").lower()
    feet_follow = 1.0 if "feet follow hands" in method else 0.0
    footless = 1.0 if "footless" in method else 0.0
    holdsets = item.get("holdsets") or []
    holdset_desc = ""
    if holdsets and isinstance(holdsets, list):
        holdset_desc = (holdsets[0].get("description") or "").lower()
    holdset_b = 1.0 if "hold set b" in holdset_desc else 0.0
    feats = np.array([
        num_moves,
        num_starts,
        num_ends,
        max_y,
        min_y,
        vert_span,
        horiz_span,
        max_dist,
        mean_dist,
        big_moves,
        user_rating,
        repeats_log,
        is_benchmark,
        feet_follow,
        footless,
        holdset_b,
    ], dtype=float)
    assert feats.shape[0] == N_EXTRA_FEATURES
    return feats


print(f"Loading data from {DATA_PATH} ...")
with DATA_PATH.open(encoding="utf-8") as f:
    payload = json.load(f)

problems: List[Dict[str, Any]] = payload["data"]

from sklearn.feature_extraction.text import TfidfVectorizer

texts: List[str] = []
y_indices: List[int] = []
extra_feats: List[np.ndarray] = []
repeats_list: List[float] = []

for item in problems:
    label_str = choose_label(item)
    if label_str not in GRADE_TO_IDX:
        continue
    tokens = build_move_tokens_from_item(item)
    text = " ".join(tokens)
    texts.append(text)
    y_indices.append(GRADE_TO_IDX[label_str])
    extra_feats.append(problem_features(item))
    repeats = float(item.get("repeats") or 0.0)
    repeats_list.append(repeats)

y = np.array(y_indices, dtype=float)
n_samples = len(texts)
print(f"Loaded {n_samples} problems after filtering.")

print("Vectorizing with TF-IDF ...")
vec = TfidfVectorizer(
    token_pattern=r"[^ ]+",
    ngram_range=(1, 2),
    min_df=2,
)
X_tfidf = vec.fit_transform(texts)
print(f"TF-IDF shape: {X_tfidf.shape}")

print("Building extra feature matrix ...")
X_extra = np.vstack(extra_feats).astype(float)
print(f"Extra features shape: {X_extra.shape}")

X = hstack([X_tfidf, X_extra])
print(f"Combined feature shape: {X.shape}")

print("Computing sample weights ...")
class_counts = np.bincount(y_indices, minlength=len(GRADE_ORDER))
base_class_weights = 1.0 / np.maximum(class_counts, 1)
base_class_weights = base_class_weights / base_class_weights.mean()
repeats_arr = np.array(repeats_list, dtype=float)
repeats_factor = np.log1p(repeats_arr)
base_weights_for_samples = base_class_weights[np.array(y_indices)]
sample_weights = base_weights_for_samples * (1.0 + 0.2 * repeats_factor)
print("Class counts:", class_counts.tolist())
print("Base class weights (approx):", np.round(base_class_weights, 3).tolist())
print("Sample weights: min=", float(sample_weights.min()),
      "max=", float(sample_weights.max()))

print("Splitting train/test ...")
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X,
    y,
    sample_weights,
    test_size=0.2,
    random_state=42,
    stratify=np.array(y_indices),
)
print(f"Train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")

print("Splitting train/val ...")
X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train,
    y_train,
    w_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train.astype(int),
)
print(f"Train(core) size: {X_tr.shape[0]}, val size: {X_val.shape[0]}")

dtrain = DMatrix(X_tr, label=y_tr, weight=w_tr)
dval = DMatrix(X_val, label=y_val, weight=w_val)

base_params: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "seed": 42,
}

lr_candidates = [0.03, 0.05, 0.1]
max_depth_candidates = [4, 6, 8]
subsample_candidates = [0.7, 0.85, 1.0]
colsample_candidates = [0.7, 0.85, 1.0]
min_child_candidates = [1, 3, 5, 10]
lambda_candidates = [0.5, 1.0, 2.0, 5.0]
alpha_candidates = [0.0, 0.1, 0.5]
gamma_candidates = [0.0, 0.5, 1.0]

max_trials = 12
patience = 3
best_score = float("inf")
best_params: Dict[str, Any] | None = None
best_rounds: int | None = None
no_improve = 0

evals = [(dval, "eval"), (dtrain, "train")]
rng = np.random.default_rng(42)

print("\nStarting hyperparameter search ...")
print(
    f"Search space: lr={lr_candidates}, "
    f"max_depth={max_depth_candidates}, "
    f"subsample={subsample_candidates}, "
    f"colsample_bytree={colsample_candidates}, "
    f"min_child_weight={min_child_candidates}, "
    f"lambda={lambda_candidates}, alpha={alpha_candidates}, gamma={gamma_candidates}"
)

for trial in range(1, max_trials + 1):
    lr = float(rng.choice(lr_candidates))
    md = int(rng.choice(max_depth_candidates))
    ss = float(rng.choice(subsample_candidates))
    cs = float(rng.choice(colsample_candidates))
    mcw = int(rng.choice(min_child_candidates))
    reg_lambda = float(rng.choice(lambda_candidates))
    reg_alpha = float(rng.choice(alpha_candidates))
    gamma = float(rng.choice(gamma_candidates))

    params = {
        **base_params,
        "learning_rate": lr,
        "max_depth": md,
        "subsample": ss,
        "colsample_bytree": cs,
        "min_child_weight": mcw,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "gamma": gamma,
    }

    print(
        f"\n[trial {trial}/{max_trials}] "
        f"lr={lr}, max_depth={md}, subsample={ss}, colsample_bytree={cs}, "
        f"min_child_weight={mcw}, lambda={reg_lambda}, alpha={reg_alpha}, gamma={gamma}"
    )

    bst = xgb_train(
        params,
        dtrain,
        num_boost_round=600,
        evals=evals,
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    score = float(bst.best_score)
    rounds = bst.best_iteration

    print(f"  -> eval rmse={score:.4f} at round {rounds}")

    if score + 1e-4 < best_score:
        best_score = score
        best_params = params
        best_rounds = rounds
        no_improve = 0
        print("  -> new best configuration")
    else:
        no_improve += 1
        print(f"  -> no improvement (streak={no_improve})")

    if no_improve >= patience:
        print("\nStopping search: no improvement for several trials.")
        break

if best_params is None or best_rounds is None:
    raise RuntimeError("Hyperparameter search failed to find a model.")

print("\nBest params found:")
print(best_params)
print(f"Best rmse: {best_score:.4f}")
print(f"Best num_boost_round: {best_rounds}")

print("\nTraining best model on train core set ...")
bst = xgb_train(
    best_params,
    dtrain,
    num_boost_round=best_rounds,
    evals=evals,
    verbose_eval=False,
)

print("Evaluating on held-out test set ...")
dtest = DMatrix(X_test, label=y_test, weight=w_test)
y_pred_reg = bst.predict(dtest)

y_pred_idx = np.rint(y_pred_reg).astype(int)
y_pred_idx = np.clip(y_pred_idx, 0, len(GRADE_ORDER) - 1)
y_test_idx = np.rint(y_test).astype(int)
y_test_idx = np.clip(y_test_idx, 0, len(GRADE_ORDER) - 1)

acc = accuracy_score(y_test_idx, y_pred_idx)
print(f"\nAccuracy (exact grade): {acc:.3f}")

grade_diff = np.abs(y_pred_idx - y_test_idx)
within_one = float((grade_diff <= 1).mean())
within_two = float((grade_diff <= 2).mean())
print(f"Within ±1 grade: {within_one:.3f}")
print(f"Within ±2 grades: {within_two:.3f}\n")

y_test_labels = [IDX_TO_GRADE[int(i)] for i in y_test_idx]
y_pred_labels = [IDX_TO_GRADE[int(i)] for i in y_pred_idx]

print("Classification report:\n")
print(classification_report(y_test_labels, y_pred_labels, labels=GRADE_ORDER))

print("Retraining best model on full data ...")
dall = DMatrix(X, label=y, weight=sample_weights)
bst_full = xgb_train(
    best_params,
    dall,
    num_boost_round=best_rounds,
    verbose_eval=False,
)

print(f"Saving artifacts to {THIS_DIR} ...")
joblib.dump(bst_full, MODEL_PATH)
joblib.dump(vec, VEC_PATH)
joblib.dump({"grade_to_idx": GRADE_TO_IDX, "idx_to_grade": IDX_TO_GRADE}, GRADE_MAP_PATH)
print("Done.")


def _build_tokens_from_moves(move_labels: List[str]) -> List[str]:
    if not move_labels:
        return []
    tokens: List[str] = []
    n = len(move_labels)
    for i, h in enumerate(move_labels):
        if i == 0:
            tokens.append(f"START_{h}")
        elif i == 1 and n > 2:
            tokens.append(f"START_{h}")
        elif i == n - 1:
            tokens.append(f"END_{h}")
        else:
            tokens.append(h)
    return tokens


def _extra_from_moves(move_labels: List[str]) -> np.ndarray:
    tokens = _build_tokens_from_moves(move_labels)
    coords = [hold_to_xy(h.replace("START_", "").replace("END_", "")) for h in tokens]
    if coords:
        xs = np.array([x for x, _ in coords], dtype=float)
        ys = np.array([y for _, y in coords], dtype=float)
    else:
        xs = np.array([0.0])
        ys = np.array([0.0])
    num_moves = float(len(coords))
    num_starts = float(min(2, len(coords)))
    num_ends = 1.0 if coords else 0.0
    max_y = float(ys.max())
    min_y = float(ys.min())
    vert_span = max_y - min_y
    horiz_span = float(xs.max() - xs.min())
    if len(coords) > 1:
        deltas = np.diff(np.c_[xs, ys], axis=0)
        dists = np.linalg.norm(deltas, axis=1)
    else:
        dists = np.array([0.0])
    max_dist = float(dists.max())
    mean_dist = float(dists.mean())
    big_moves = float((dists >= 4.0).sum())
    user_rating = 0.0
    repeats_log = 0.0
    is_benchmark = 0.0
    feet_follow = 1.0
    footless = 0.0
    holdset_b = 0.0
    x_extra = np.array([[
        num_moves,
        num_starts,
        num_ends,
        max_y,
        min_y,
        vert_span,
        horiz_span,
        max_dist,
        mean_dist,
        big_moves,
        user_rating,
        repeats_log,
        is_benchmark,
        feet_follow,
        footless,
        holdset_b,
    ]], dtype=float)
    assert x_extra.shape[1] == N_EXTRA_FEATURES
    return tokens, x_extra


def predict_grade(move_labels: List[str]) -> str:
    bst_loaded = joblib.load(MODEL_PATH)
    vec_loaded = joblib.load(VEC_PATH)
    grade_map = joblib.load(GRADE_MAP_PATH)
    idx_to_grade_local = grade_map["idx_to_grade"]
    tokens, x_extra = _extra_from_moves(move_labels)
    text = " ".join(tokens)
    X_tfidf = vec_loaded.transform([text])
    X_input = hstack([X_tfidf, x_extra])
    dinput = DMatrix(X_input)
    y_pred_reg = bst_loaded.predict(dinput)[0]
    idx = int(np.rint(y_pred_reg))
    idx = max(0, min(idx, len(idx_to_grade_local) - 1))
    return idx_to_grade_local[idx]
