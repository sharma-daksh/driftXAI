import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "data/Training Dataset.arff"

data, meta = arff.loadarff(DATA_PATH)

df = pd.DataFrame(data)


# ==========================================
# 2. CONVERT DATA TO NUMERIC
# ==========================================

for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    df[column] = pd.to_numeric(df[column])


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Result", axis=1)
y = df["Result"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== DATA SPLIT ==========")
print("Total samples :", len(df))
print("Training      :", len(X_train))
print("Testing       :", len(X_test))


# ==========================================
# 5. TRAIN MODEL V1
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL V1 ==========")
print("Accuracy:", round(accuracy, 4))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Phishing (-1)", "Legitimate (1)"]
))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))