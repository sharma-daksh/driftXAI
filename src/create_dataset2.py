import pandas as pd
import numpy as np
from scipy.io import arff


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/Training Dataset.arff"
OUTPUT_PATH = "data/dataset2_concept_drift.csv"

# Percentage of Dataset 2 following the new concept
DRIFT_SHARE = 0.80


# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================

data, meta = arff.loadarff(DATA_PATH)

df = pd.DataFrame(data)


# ==========================================
# 2. CONVERT VALUES TO NUMERIC
# ==========================================

for column in df.columns:
    df[column] = df[column].map(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    df[column] = pd.to_numeric(df[column])


# ==========================================
# 3. SEPARATE FEATURES AND ORIGINAL TARGET
# ==========================================

X = df.drop(columns=["Result"])
original_y = df["Result"].copy()


# ==========================================
# 4. DEFINE A NEW CONCEPT
# ==========================================
#
# Dataset 1 learned from the original target.
#
# Dataset 2 represents an environment where
# different website characteristics become more
# influential.
#
# IMPORTANT:
# This is a controlled synthetic concept-drift
# experiment.
#

weights = {
    "Links_in_tags": 3.0,
    "Prefix_Suffix": 2.5,
    "Request_URL": 2.0,
    "having_Sub_Domain": 1.5,
    "web_traffic": 1.0
}


# Calculate new concept score
new_concept_score = np.zeros(len(df))

for feature, weight in weights.items():
    new_concept_score += X[feature] * weight


# Add small noise so the new concept is not
# a perfectly deterministic rule
score_rng = np.random.RandomState(42)

new_concept_score += score_rng.normal(
    loc=0,
    scale=1.0,
    size=len(df)
)


# Use median so the new concept has
# approximately balanced classes
threshold = np.median(new_concept_score)

new_concept_target = np.where(
    new_concept_score >= threshold,
    1,
    -1
)


# ==========================================
# 5. CREATE GRADUAL CONCEPT DRIFT
# ==========================================

transition_rng = np.random.RandomState(100)

transition_mask = (
    transition_rng.rand(len(df)) < DRIFT_SHARE
)


# Start with original concept
dataset2 = df.copy()


# 80% follows new concept,
# 20% retains original concept
dataset2["Result"] = np.where(
    transition_mask,
    new_concept_target,
    original_y
)


# ==========================================
# 6. SAVE DATASET 2
# ==========================================

dataset2.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# 7. REPORT
# ==========================================

print("\n==========================================")
print("       DATASET 2 - CONCEPT DRIFT")
print("==========================================")

print("\nDataset size:", len(dataset2))

print(
    "New-concept transition:",
    round(DRIFT_SHARE * 100, 2),
    "%"
)

actual_changed = (
    dataset2["Result"] != original_y
).sum()

print(
    "Actual target values changed:",
    actual_changed
)

print(
    "Actual target change rate:",
    round(
        actual_changed / len(dataset2) * 100,
        2
    ),
    "%"
)


# ==========================================
# 8. TARGET DISTRIBUTIONS
# ==========================================

print("\n========== DATASET 1 TARGET ==========")

print(
    original_y.value_counts()
)


print("\n========== NEW CONCEPT TARGET ==========")

print(
    pd.Series(new_concept_target).value_counts()
)


print("\n========== DATASET 2 TARGET ==========")

print(
    dataset2["Result"].value_counts()
)


# ==========================================
# 9. IMPORTANT RELATIONSHIP CHANGE
# ==========================================

print("\n==========================================")
print("     SSL TARGET RELATIONSHIP CHANGE")
print("==========================================")


print("\nDataset 1:")

print(
    pd.crosstab(
        df["SSLfinal_State"],
        original_y,
        normalize="index"
    ).round(3)
)


print("\nDataset 2:")

print(
    pd.crosstab(
        dataset2["SSLfinal_State"],
        dataset2["Result"],
        normalize="index"
    ).round(3)
)


# ==========================================
# 10. NEW CONCEPT FEATURES
# ==========================================

print("\n==========================================")
print("       NEW CONCEPT DRIVERS")
print("==========================================")

for feature, weight in weights.items():
    print(
        f"{feature:<25} weight = {weight}"
    )


print("\nSaved to:")
print(OUTPUT_PATH)