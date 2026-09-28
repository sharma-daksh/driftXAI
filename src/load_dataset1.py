import pandas as pd
from scipy.io import arff


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "data/Training Dataset.arff"

data, meta = arff.loadarff(DATA_PATH)

df = pd.DataFrame(data)


# ==========================================
# 2. CONVERT BYTES/STRINGS TO NUMBERS
# ==========================================

for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    df[column] = pd.to_numeric(df[column])


# ==========================================
# 3. BASIC DATASET INFORMATION
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Result"].value_counts())


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Result", axis=1)
y = df["Result"]

print("\n========== FEATURES ==========")
print("Number of features:", X.shape[1])

print("\n========== TARGET ==========")
print("Target column:", y.name)


# ==========================================
# 5. DISPLAY FIRST 5 ROWS
# ==========================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())