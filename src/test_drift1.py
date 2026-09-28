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
# 1. LOAD ORIGINAL DATASET
# ==========================================

DATA_PATH = "data/Training Dataset.arff"

data, meta = arff.loadarff(DATA_PATH)

df = pd.DataFrame(data)


# ==========================================
# 2. CONVERT TO NUMERIC
# ==========================================

for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    df[column] = pd.to_numeric(df[column])


# ==========================================
# 3. CREATE SAME TRAIN/TEST SPLIT
# ==========================================

X = df.drop("Result", axis=1)
y = df["Result"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. TRAIN MODEL V1
# ==========================================

model_v1 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_v1.fit(X_train, y_train)


# ==========================================
# 5. LOAD DATASET 2
# ==========================================

dataset2 = pd.read_csv(
    "data/dataset2_drift.csv"
)

X_drift = dataset2.drop("Result", axis=1)
y_drift = dataset2["Result"]


# ==========================================
# 6. PREDICT DATASET 2 USING OLD MODEL
# ==========================================

y_drift_pred = model_v1.predict(X_drift)


# ==========================================
# 7. EVALUATE OLD MODEL ON NEW DATA
# ==========================================

accuracy_drift = accuracy_score(
    y_drift,
    y_drift_pred
)


print("\n==========================================")
print("       MODEL V1 UNDER CONCEPT DRIFT")
print("==========================================")

print("\nOriginal Model V1 Accuracy:")
print("97.42%")

print("\nModel V1 Accuracy on Dataset 2:")
print(round(accuracy_drift * 100, 2), "%")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_drift,
        y_drift_pred,
        target_names=[
            "Phishing (-1)",
            "Legitimate (1)"
        ]
    )
)


print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_drift,
        y_drift_pred
    )
)


# ==========================================
# 8. ACCURACY DROP
# ==========================================

original_accuracy = accuracy_score(
    y_test,
    model_v1.predict(X_test)
)

accuracy_drop = original_accuracy - accuracy_drift


print("\n========== DRIFT IMPACT ==========")

print(
    "Original Accuracy:",
    round(original_accuracy * 100, 2),
    "%"
)

print(
    "Drift Accuracy:",
    round(accuracy_drift * 100, 2),
    "%"
)

print(
    "Accuracy Drop:",
    round(accuracy_drop * 100, 2),
    "percentage points"
)