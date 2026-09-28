import pandas as pd
import numpy as np
import shap

from scipy.io import arff
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# CONFIGURATION
# =========================================================

SHAP_SAMPLE_SIZE = 300
TOP_K = 5
RANDOM_STATE = 42


# =========================================================
# 1. LOAD DATASET 1
# =========================================================

data, meta = arff.loadarff(
    "data/Training Dataset.arff"
)

df1 = pd.DataFrame(data)

for column in df1.columns:
    df1[column] = df1[column].map(
        lambda x: x.decode("utf-8")
        if isinstance(x, bytes)
        else x
    )

    df1[column] = pd.to_numeric(df1[column])


# =========================================================
# 2. LOAD DATASET 2
# =========================================================

df2 = pd.read_csv(
    "data/dataset2_concept_drift.csv"
)


# =========================================================
# 3. FEATURES / TARGET
# =========================================================

feature_columns = [
    column for column in df1.columns
    if column != "Result"
]

X1 = df1[feature_columns]
y1 = df1["Result"]

X2 = df2[feature_columns]
y2 = df2["Result"]


# =========================================================
# 4. SAME SPLIT USED THROUGHOUT PROJECT
# =========================================================

train_indices, test_indices = train_test_split(
    df1.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y1
)

X1_train = X1.loc[train_indices]
y1_train = y1.loc[train_indices]

X2_train = X2.loc[train_indices]
y2_train = y2.loc[train_indices]

X2_test = X2.loc[test_indices]
y2_test = y2.loc[test_indices]


# =========================================================
# 5. TRAIN NORMAL MODEL V2
# =========================================================

model_v2 = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model_v2.fit(
    X2_train,
    y2_train
)


# =========================================================
# 6. RANDOMIZE TRAINING LABELS
# =========================================================

rng = np.random.RandomState(
    RANDOM_STATE
)

randomized_y = y2_train.to_numpy().copy()

rng.shuffle(randomized_y)

randomized_y = pd.Series(
    randomized_y,
    index=y2_train.index,
    name="Result"
)


# =========================================================
# 7. TRAIN RANDOMIZED MODEL
# =========================================================

randomized_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

randomized_model.fit(
    X2_train,
    randomized_y
)


# =========================================================
# 8. TEST RANDOMIZED MODEL
# =========================================================

randomized_predictions = randomized_model.predict(
    X2_test
)

randomized_accuracy = accuracy_score(
    y2_test,
    randomized_predictions
)


print("\n==========================================")
print("       RANDOMIZATION TEST")
print("==========================================")

print(
    "Randomized-model accuracy on real Dataset 2:",
    round(randomized_accuracy * 100, 2),
    "%"
)


# =========================================================
# 9. SHAP SAMPLE
# =========================================================

X_eval = X2_test.iloc[
    :SHAP_SAMPLE_SIZE
]


# =========================================================
# 10. SHAP
# =========================================================

print("\nCalculating SHAP values...")

explainer_normal = shap.TreeExplainer(
    model_v2
)

explainer_randomized = shap.TreeExplainer(
    randomized_model
)

shap_normal = explainer_normal.shap_values(
    X_eval
)

shap_randomized = (
    explainer_randomized.shap_values(
        X_eval
    )
)


# =========================================================
# 11. HANDLE SHAP OUTPUT
# =========================================================

def get_positive_class_shap(values):

    if isinstance(values, list):
        return values[1]

    if len(values.shape) == 3:
        return values[:, :, 1]

    return values


shap_normal = get_positive_class_shap(
    shap_normal
)

shap_randomized = get_positive_class_shap(
    shap_randomized
)


# =========================================================
# 12. GLOBAL IMPORTANCE
# =========================================================

normal_importance = np.abs(
    shap_normal
).mean(axis=0)

randomized_importance = np.abs(
    shap_randomized
).mean(axis=0
)


importance = pd.DataFrame({
    "Feature": feature_columns,
    "Normal_V2": normal_importance,
    "Randomized_Model": randomized_importance
})


importance["Normal_Rank"] = (
    importance["Normal_V2"]
    .rank(
        ascending=False,
        method="min"
    )
)

importance["Randomized_Rank"] = (
    importance["Randomized_Model"]
    .rank(
        ascending=False,
        method="min"
    )
)


# =========================================================
# 13. RANK CORRELATION
# =========================================================

rank_correlation, p_value = spearmanr(
    normal_importance,
    randomized_importance
)


# =========================================================
# 14. TOP-K OVERLAP
# =========================================================

normal_top_features = set(
    importance
    .sort_values(
        "Normal_V2",
        ascending=False
    )
    .head(TOP_K)["Feature"]
)

randomized_top_features = set(
    importance
    .sort_values(
        "Randomized_Model",
        ascending=False
    )
    .head(TOP_K)["Feature"]
)

top_overlap = len(
    normal_top_features
    & randomized_top_features
)


# =========================================================
# 15. DISPLAY RESULTS
# =========================================================

print("\n==========================================")
print("       SHAP RANDOMIZATION RESULTS")
print("==========================================")

print(
    "\nTop features - Normal Model V2:"
)

print(
    importance
    .sort_values(
        "Normal_V2",
        ascending=False
    )
    [
        [
            "Feature",
            "Normal_V2"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print(
    "\nTop features - Randomized-label Model:"
)

print(
    importance
    .sort_values(
        "Randomized_Model",
        ascending=False
    )
    [
        [
            "Feature",
            "Randomized_Model"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print(
    "\nSpearman correlation:",
    round(rank_correlation, 4)
)

print(
    "p-value:",
    round(p_value, 6)
)

print(
    f"Top-{TOP_K} feature overlap:",
    f"{top_overlap}/{TOP_K}"
)


# =========================================================
# 16. SAVE
# =========================================================

importance.to_csv(
    "data/randomization_results.csv",
    index=False
)

summary = pd.DataFrame({
    "Randomized_Model_Accuracy": [
        randomized_accuracy
    ],
    "SHAP_Rank_Correlation": [
        rank_correlation
    ],
    "Top5_Overlap": [
        top_overlap
    ]
})

summary.to_csv(
    "data/randomization_summary.csv",
    index=False
)

print("\nSaved:")
print("data/randomization_results.csv")
print("data/randomization_summary.csv")