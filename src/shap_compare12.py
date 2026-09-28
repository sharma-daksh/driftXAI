import pandas as pd
import numpy as np
import shap

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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
# 3. FEATURES AND TARGET
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
# 4. SAME TRAIN / TEST SPLIT
# =========================================================

train_indices, test_indices = train_test_split(
    df1.index,
    test_size=0.20,
    random_state=42,
    stratify=y1
)


X1_train = X1.loc[train_indices]
y1_train = y1.loc[train_indices]

X2_train = X2.loc[train_indices]
y2_train = y2.loc[train_indices]

X2_test = X2.loc[test_indices]
y2_test = y2.loc[test_indices]
# Use a smaller sample for SHAP analysis only.
# Model evaluation still uses all test samples.
SHAP_SAMPLE_SIZE = 300

X2_shap = X2_test.iloc[:SHAP_SAMPLE_SIZE]

# =========================================================
# 5. TRAIN MODEL V1
# =========================================================

model_v1 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_v1.fit(
    X1_train,
    y1_train
)


# =========================================================
# 6. TRAIN MODEL V2
# =========================================================

model_v2 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_v2.fit(
    X2_train,
    y2_train
)


# =========================================================
# 7. SHAP EXPLAINERS
# =========================================================

explainer_v1 = shap.TreeExplainer(model_v1)
explainer_v2 = shap.TreeExplainer(model_v2)


shap_v1 = explainer_v1.shap_values(X2_shap)
shap_v2 = explainer_v2.shap_values(X2_shap)


# =========================================================
# 8. HANDLE SHAP OUTPUT FORMAT
# =========================================================

def get_positive_class_shap(shap_values):

    if isinstance(shap_values, list):

        # For classes [-1, 1], index 1 = class +1
        return shap_values[1]

    if len(shap_values.shape) == 3:

        # Last dimension contains class outputs
        return shap_values[:, :, 1]

    return shap_values


shap_v1 = get_positive_class_shap(shap_v1)
shap_v2 = get_positive_class_shap(shap_v2)


# =========================================================
# 9. GLOBAL SHAP IMPORTANCE
# =========================================================

importance_v1 = pd.DataFrame({
    "Feature": feature_columns,
    "V1_Importance": np.abs(shap_v1).mean(axis=0)
})

importance_v2 = pd.DataFrame({
    "Feature": feature_columns,
    "V2_Importance": np.abs(shap_v2).mean(axis=0)
})


comparison = importance_v1.merge(
    importance_v2,
    on="Feature"
)


comparison["Change"] = (
    comparison["V2_Importance"]
    - comparison["V1_Importance"]
)

comparison["Absolute_Change"] = (
    abs(comparison["Change"])
)


comparison = comparison.sort_values(
    by="V2_Importance",
    ascending=False
)


# =========================================================
# 10. PRINT GLOBAL COMPARISON
# =========================================================

print("\n==========================================")
print("       GLOBAL SHAP COMPARISON")
print("==========================================")

print("\nTop features used by Model V1 on Dataset 2:")

print(
    comparison
    .sort_values(
        by="V1_Importance",
        ascending=False
    )
    [["Feature", "V1_Importance"]]
    .head(10)
    .to_string(index=False)
)


print("\nTop features used by Model V2 on Dataset 2:")

print(
    comparison[
        [
            "Feature",
            "V1_Importance",
            "V2_Importance",
            "Change"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# =========================================================
# 11. BIGGEST SHAP CHANGES
# =========================================================

print("\n==========================================")
print("       BIGGEST SHAP CHANGES")
print("==========================================")


print(
    comparison
    .sort_values(
        by="Absolute_Change",
        ascending=False
    )
    [
        [
            "Feature",
            "V1_Importance",
            "V2_Importance",
            "Change"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# =========================================================
# 12. LOCAL EXPLANATION
# =========================================================
#
# SAME TEST SAMPLE is used for both models.
#

sample_position = 0

sample = X2_shap.iloc[
    [sample_position]
]

sample_index = X2_test.index[
    sample_position
]


# Predictions
prediction_v1 = model_v1.predict(sample)[0]
prediction_v2 = model_v2.predict(sample)[0]


def label(value):

    if value == -1:
        return "PHISHING"

    return "LEGITIMATE"


print("\n==========================================")
print("          LOCAL SHAP COMPARISON")
print("==========================================")

print("\nOriginal row index:", sample_index)

print(
    "Actual Dataset 2 label:",
    label(y2_test.loc[sample_index])
)

print(
    "Model V1 prediction:",
    label(prediction_v1)
)

print(
    "Model V2 prediction:",
    label(prediction_v2)
)


# =========================================================
# 13. LOCAL V1
# =========================================================

local_v1 = pd.DataFrame({
    "Feature": feature_columns,
    "Value": sample.iloc[0].values,
    "V1_SHAP": shap_v1[sample_position]
})

local_v1["Abs_SHAP"] = abs(
    local_v1["V1_SHAP"]
)

local_v1 = local_v1.sort_values(
    by="Abs_SHAP",
    ascending=False
)


# =========================================================
# 14. LOCAL V2
# =========================================================

local_v2 = pd.DataFrame({
    "Feature": feature_columns,
    "Value": sample.iloc[0].values,
    "V2_SHAP": shap_v2[sample_position]
})

local_v2["Abs_SHAP"] = abs(
    local_v2["V2_SHAP"]
)

local_v2 = local_v2.sort_values(
    by="Abs_SHAP",
    ascending=False
)


print("\n========== V1 LOCAL EXPLANATION ==========")

print(
    local_v1[
        [
            "Feature",
            "Value",
            "V1_SHAP"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\n========== V2 LOCAL EXPLANATION ==========")

print(
    local_v2[
        [
            "Feature",
            "Value",
            "V2_SHAP"
        ]
    ]
    .head(10)
    .to_string(index=False)
)