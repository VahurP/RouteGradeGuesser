from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack
from xgboost import DMatrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "training"

MODEL_PATH = ARTIFACTS_DIR / "grade_model.joblib"
VEC_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
GRADE_MAP_PATH = ARTIFACTS_DIR / "grade_map.joblib"

COLUMNS = list("ABCDEFGHIJK")
N_EXTRA_FEATURES = 16


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


def _extra_from_moves(move_labels: List[str]) -> Tuple[List[str], np.ndarray]:
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

    x_extra = np.array(
        [
            [
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
            ]
        ],
        dtype=float,
    )

    assert x_extra.shape[1] == N_EXTRA_FEATURES
    return tokens, x_extra


class GradeModel:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        print(f"Loading model artifacts from {ARTIFACTS_DIR}...")
        self.bst = joblib.load(MODEL_PATH)
        self.vec = joblib.load(VEC_PATH)
        grade_map: Dict[str, Dict[Any, Any]] = joblib.load(GRADE_MAP_PATH)
        self.idx_to_grade: Dict[int, str] = grade_map["idx_to_grade"]
        self.num_grades = len(self.idx_to_grade)
        print("Model artifacts loaded.")

    def predict(self, moves: List[str]) -> Tuple[str, List[str]]:
        if not moves:
            raise ValueError("No moves provided")

        tokens, x_extra = _extra_from_moves(moves)
        text = " ".join(tokens)
        X_tfidf = self.vec.transform([text])
        X_input = hstack([X_tfidf, x_extra])
        dinput = DMatrix(X_input)

        y_pred_reg = float(self.bst.predict(dinput)[0])
        idx = int(np.rint(y_pred_reg))
        idx = max(0, min(idx, self.num_grades - 1))

        grade = self.idx_to_grade[idx]
        return grade, moves
