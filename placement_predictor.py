import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["ExtracurricularActivities", "PlacementTraining"]
DROP_COLS = ["StudentID", "PlacementStatus"]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
plc_data = pd.read_csv("placementdata .csv")

X = plc_data.drop(columns=DROP_COLS)
y = plc_data["PlacementStatus"].map({"NotPlaced": 0, "Placed": 1})

# ---------------------------------------------------------------------------
# 2. Train/test split — stratified so both splits keep the same class ratio
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Preprocessing — one-hot encode the two categorical columns only.
#    Numeric columns pass through untouched 
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="if_binary"), CATEGORICAL_COLS)],
    remainder="passthrough",
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

# ---------------------------------------------------------------------------
# 4. Hyperparameter tuning — grid search with cross-validation, training
#    data only, scored on F1
# ---------------------------------------------------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 4],
}

search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, refit=True
)
search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
print(f"Best params: {search.best_params_}")
print(f"Best CV F1 (train folds): {search.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 5. Evaluate ONLY on the held-out test set — never on rows the model
#    trained on.
# ---------------------------------------------------------------------------
y_pred = best_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Held-out test set performance ---")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")
print(f"Confusion matrix:\n{cm}")

# ---------------------------------------------------------------------------
# 6. Save artifacts:
# ---------------------------------------------------------------------------
with open("pipeline.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

with open("test_split.pkl", "wb") as f:
    pickle.dump({"X_test": X_test, "y_test": y_test}, f)

# ---------------------------------------------------------------------------
# 7. Sanity-check prediction on a single made-up student (no StudentID needed)
# ---------------------------------------------------------------------------
sample = pd.DataFrame({
    "CGPA": [9.79],
    "Internships": [0],
    "Projects": [2],
    "Workshops/Certifications": [3],
    "AptitudeTestScore": [80],
    "SoftSkillsRating": [3],
    "ExtracurricularActivities": ["Yes"],
    "PlacementTraining": ["Yes"],
    "SSC_Marks": [85],
    "HSC_Marks": [82],
})

proba = best_pipeline.predict_proba(sample)[0][1]  # P(Placed)
status = "Placed" if proba >= 0.5 else "Not Placed"
print(f"\nSample prediction: {status} (P(Placed) = {proba:.2f})")