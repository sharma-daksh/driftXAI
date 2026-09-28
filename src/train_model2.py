import pandas as pd

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET 1
# ==========================================

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


# ==========================================
# 2. LOAD DATASET 2
# ==========================================

df2 = pd.read_csv(
    "data/dataset2_concept_drift.csv"
)


# ==========================================
# 3. VERIFY FEATURES
# ==========================================

feature_columns = [
    column for column in df1.columns
    if column != "Result"
]


X1 = df1[feature_columns]
y1 = df1["Result"]

X2 = df2[feature_columns]
y2 = df2["Result"]


# ==========================================
# 4. SAME TRAIN/TEST INDICES
# ==========================================
#
# IMPORTANT:
# We create the split ONCE and use the
# exact same rows for V1 and V2.
#

train_indices, test_indices = train_test_split(
    df1.index,
    test_size=0.20,
    random_state=42,
    stratify=y1
)


print("\n==========================================")
print("            FAIR DATA SPLIT")
print("==========================================")

print("Training samples:", len(train_indices))
print("Testing samples :", len(test_indices))


# ==========================================
# 5. TRAIN MODEL V1
# ==========================================

model_v1 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_v1.fit(
    X1.loc[train_indices],
    y1.loc[train_indices]
)


# ==========================================
# 6. TRAIN MODEL V2
# ==========================================
#
# Same feature rows,
# but using the new concept labels.
#

model_v2 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_v2.fit(
    X2.loc[train_indices],
    y2.loc[train_indices]
)


# ==========================================
# 7. PREDICTIONS
# ==========================================

# V1 on original concept
v1_d1_pred = model_v1.predict(
    X1.loc[test_indices]
)

# V1 on drifted concept
v1_d2_pred = model_v1.predict(
    X2.loc[test_indices]
)

# V2 on drifted concept
v2_d2_pred = model_v2.predict(
    X2.loc[test_indices]
)


# ==========================================
# 8. ACCURACIES
# ==========================================

v1_d1_accuracy = accuracy_score(
    y1.loc[test_indices],
    v1_d1_pred
)

v1_d2_accuracy = accuracy_score(
    y2.loc[test_indices],
    v1_d2_pred
)

v2_d2_accuracy = accuracy_score(
    y2.loc[test_indices],
    v2_d2_pred
)


# ==========================================
# 9. RESULTS
# ==========================================

print("\n==========================================")
print("              MODEL COMPARISON")
print("==========================================")

print(
    "\nModel V1 → Dataset 1:",
    round(v1_d1_accuracy * 100, 2),
    "%"
)

print(
    "Model V1 → Dataset 2:",
    round(v1_d2_accuracy * 100, 2),
    "%"
)

print(
    "Model V2 → Dataset 2:",
    round(v2_d2_accuracy * 100, 2),
    "%"
)


# ==========================================
# 10. DRIFT IMPACT
# ==========================================

accuracy_drop = (
    v1_d1_accuracy - v1_d2_accuracy
)

adaptation_gain = (
    v2_d2_accuracy - v1_d2_accuracy
)


print("\n==========================================")
print("              DRIFT IMPACT")
print("==========================================")

print(
    "Accuracy drop after drift:",
    round(
        accuracy_drop * 100,
        2
    ),
    "percentage points"
)

print(
    "Recovery after adaptation:",
    round(
        adaptation_gain * 100,
        2
    ),
    "percentage points"
)


# ==========================================
# 11. V1 ON DATASET 2
# ==========================================

print("\n==========================================")
print("       MODEL V1 ON DATASET 2")
print("==========================================")

print(
    classification_report(
        y2.loc[test_indices],
        v1_d2_pred,
        target_names=[
            "Phishing (-1)",
            "Legitimate (1)"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y2.loc[test_indices],
        v1_d2_pred
    )
)


# ==========================================
# 12. V2 ON DATASET 2
# ==========================================

print("\n==========================================")
print("       MODEL V2 ON DATASET 2")
print("==========================================")

print(
    classification_report(
        y2.loc[test_indices],
        v2_d2_pred,
        target_names=[
            "Phishing (-1)",
            "Legitimate (1)"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y2.loc[test_indices],
        v2_d2_pred
    )
)