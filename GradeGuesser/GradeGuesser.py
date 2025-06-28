import json
import random
import joblib
import numpy as np
from xgboost import DMatrix

PATH = r"C:\Users\vahurpaist\Downloads\problems_2023_01_30\problems MoonBoard 2016 .json"
def grade_toint(grade):
    grades = ["6B", "6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A", "8A+", "8B", "8B+"]
    return grades.index(grade)
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


# ---------- 6. Example usage ----------
if __name__ == "__main__":
    # Load artifacts
    clf = joblib.load("grade_model.joblib")
    vec = joblib.load("vectorizer.joblib")
    le  = joblib.load("label_encoder.joblib")

    # Load raw dataset again
    with open(PATH, encoding="utf-8") as f:
        payload = json.load(f)
    n = 10000
    samples = random.sample(payload["data"], n)
    count = 0
    for i, item in enumerate(samples, 1):
        move_labels = [m["description"] for m in item["moves"]]
        true_grade = item["grade"]
        predicted_grade = predict_grade(move_labels)
        if (abs(grade_toint(true_grade) - grade_toint(predicted_grade)) <= 1):
            count += 1
    print(count/n)

    andres_moves = ["K4", "H5", "F5", "I7", "F8", "H10", "J11", "E13", "C16", "D18", "F18"]
    andres_grade = "6b"
    predicted_grade = predict_grade(andres_moves)
    print("   Andres Moves:           ", andres_moves)
    print("  Andres actual grade:    ", andres_grade)
    print("  Predicted grade: ", predicted_grade)