import pandas as pd
import numpy as np
import shap

from scipy.io import arff
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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
# 4. SAME TRAIN / TEST SPLIT
# =========================================================

train_indices, test_indices = train_test_split(
    df1.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y1
)


# =========================================================
# 5. TRAIN MODEL V1
# =========================================================

model_v1 = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model_v1.fit(
    X1.loc[train_indices],
    y1.loc[train_indices]
)


# =========================================================
# 6. TRAIN MODEL V2
# =========================================================

model_v2 = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model_v2.fit(
    X2.loc[train_indices],
    y2.loc[train_indices]
)


# =========================================================
# 7. USE SAME 300 TEST SAMPLES
# =========================================================

X2_test = X2.loc[test_indices]

X_eval = X2_test.iloc[
    :SHAP_SAMPLE_SIZE
].copy()

print("\n==========================================")
print("       XAI EVALUATION DATA")
print("==========================================")

print(
    "Total Dataset 2 test samples:",
    len(X2_test)
)

print(
    "Samples used for XAI tests:",
    len(X_eval)
)


# =========================================================
# 8. CREATE SHAP EXPLAINERS
# =========================================================

explainer_v1 = shap.TreeExplainer(model_v1)
explainer_v2 = shap.TreeExplainer(model_v2)


shap_v1 = explainer_v1.shap_values(X_eval)
shap_v2 = explainer_v2.shap_values(X_eval)


# =========================================================
# 9. HANDLE SHAP OUTPUT FORMAT
# =========================================================

def get_positive_class_shap(shap_values):

    if isinstance(shap_values, list):
        return shap_values[1]

    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]

    return shap_values


shap_v1 = get_positive_class_shap(shap_v1)
shap_v2 = get_positive_class_shap(shap_v2)


# =========================================================
# 10. BASELINE / OCCLUSION VALUES
# =========================================================
#
# We replace an occluded feature with the most common
# value seen in the corresponding training data.
#

baseline_v1 = {}

for feature in feature_columns:
    baseline_v1[feature] = (
        X1.loc[train_indices, feature]
        .mode()
        .iloc[0]
    )


baseline_v2 = {}

for feature in feature_columns:
    baseline_v2[feature] = (
        X2.loc[train_indices, feature]
        .mode()
        .iloc[0]
    )


# =========================================================
# HELPER: GET PROBABILITY OF A PARTICULAR CLASS
# =========================================================

def class_probability(model, X, class_label):

    probabilities = model.predict_proba(X)

    class_index = list(
        model.classes_
    ).index(class_label)

    return probabilities[:, class_index]


# =========================================================
# FIDELITY TEST
# =========================================================

def run_fidelity_test(
    model,
    shap_values,
    baseline_values,
    model_name
):

    top_feature_drops = []
    random_feature_drops = []

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    feature_count = len(feature_columns)

    for i in range(len(X_eval)):

        sample = X_eval.iloc[[i]]

        # Original prediction
        predicted_class = model.predict(
            sample
        )[0]

        original_probability = class_probability(
            model,
            sample,
            predicted_class
        )[0]

        # -------------------------------------------------
        # Top-K SHAP features
        # -------------------------------------------------

        absolute_shap = np.abs(
            shap_values[i]
        )

        top_indices = np.argsort(
            absolute_shap
        )[-TOP_K:]

        top_features = [
            feature_columns[j]
            for j in top_indices
        ]

        top_occluded = sample.copy()

        for feature in top_features:
            top_occluded.loc[
                top_occluded.index,
                feature
            ] = baseline_values[feature]

        top_probability = class_probability(
            model,
            top_occluded,
            predicted_class
        )[0]

        top_drop = abs(
            original_probability
            - top_probability
        )

        top_feature_drops.append(
            top_drop
        )

        # -------------------------------------------------
        # Random-K control
        # -------------------------------------------------

        random_indices = rng.choice(
            feature_count,
            size=TOP_K,
            replace=False
        )

        random_features = [
            feature_columns[j]
            for j in random_indices
        ]

        random_occluded = sample.copy()

        for feature in random_features:
            random_occluded.loc[
                random_occluded.index,
                feature
            ] = baseline_values[feature]

        random_probability = class_probability(
            model,
            random_occluded,
            predicted_class
        )[0]

        random_drop = abs(
            original_probability
            - random_probability
        )

        random_feature_drops.append(
            random_drop
        )

    mean_top_drop = np.mean(
        top_feature_drops
    )

    mean_random_drop = np.mean(
        random_feature_drops
    )

    fidelity_ratio = (
        mean_top_drop /
        mean_random_drop
        if mean_random_drop > 0
        else np.nan
    )

    print("\n==========================================")
    print(
        f"          FIDELITY - {model_name}"
    )
    print("==========================================")

    print(
        "Mean probability change - "
        "Top SHAP features:",
        round(mean_top_drop, 4)
    )

    print(
        "Mean probability change - "
        "Random features:",
        round(mean_random_drop, 4)
    )

    print(
        "Fidelity ratio:",
        round(fidelity_ratio, 4)
    )

    return fidelity_ratio


# =========================================================
# FEATURE OCCLUSION TEST
# =========================================================

def run_occlusion_test(
    model,
    shap_values,
    baseline_values,
    model_name
):

    # SHAP global importance
    shap_importance = np.abs(
        shap_values
    ).mean(axis=0)

    # Original predictions for all samples
    original_predictions = model.predict(
        X_eval
    )

    original_probabilities = np.max(
        model.predict_proba(X_eval),
        axis=1
    )

    occlusion_impacts = []

    # -----------------------------------------------------
    # Occlude one feature at a time
    # -----------------------------------------------------

    for feature in feature_columns:

        modified = X_eval.copy()

        # Replace this feature with its baseline value
        modified[feature] = baseline_values[feature]

        # Predict ALL 300 samples at once
        modified_probabilities = np.max(
            model.predict_proba(modified),
            axis=1
        )

        # Measure prediction confidence change
        impacts = np.abs(
            original_probabilities
            - modified_probabilities
        )

        mean_impact = np.mean(
            impacts
        )

        occlusion_impacts.append(
            mean_impact
        )

    occlusion_importance = np.array(
        occlusion_impacts
    )

    # -----------------------------------------------------
    # Compare SHAP ranking with occlusion ranking
    # -----------------------------------------------------

    correlation, p_value = spearmanr(
        shap_importance,
        occlusion_importance
    )

    results = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP_Importance": shap_importance,
        "Occlusion_Impact": occlusion_importance
    })

    results["Rank_SHAP"] = (
        results["SHAP_Importance"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    results["Rank_Occlusion"] = (
        results["Occlusion_Impact"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    results = results.sort_values(
        by="SHAP_Importance",
        ascending=False
    )

    print("\n==========================================")
    print(
        f"      FEATURE OCCLUSION - {model_name}"
    )
    print("==========================================")

    print(
        results.head(10).to_string(
            index=False
        )
    )

    print(
        "\nSpearman correlation between "
        "|SHAP| and occlusion impact:",
        round(correlation, 4)
    )

    print(
        "p-value:",
        round(p_value, 6)
    )

    return results, correlation


# =========================================================
# RUN FOR MODEL V1
# =========================================================

fidelity_v1 = run_fidelity_test(
    model_v1,
    shap_v1,
    baseline_v1,
    "MODEL V1"
)

occlusion_v1, correlation_v1 = run_occlusion_test(
    model_v1,
    shap_v1,
    baseline_v1,
    "MODEL V1"
)


# =========================================================
# RUN FOR MODEL V2
# =========================================================

fidelity_v2 = run_fidelity_test(
    model_v2,
    shap_v2,
    baseline_v2,
    "MODEL V2"
)

occlusion_v2, correlation_v2 = run_occlusion_test(
    model_v2,
    shap_v2,
    baseline_v2,
    "MODEL V2"
)


# =========================================================
# FINAL COMPARISON
# =========================================================

print("\n==========================================")
print("      XAI QUALITY COMPARISON")
print("==========================================")

print("\n                 Model V1       Model V2")

print(
    "Fidelity ratio :",
    f"{fidelity_v1:.4f}",
    "       ",
    f"{fidelity_v2:.4f}"
)

print(
    "Occlusion corr.:",
    f"{correlation_v1:.4f}",
    "       ",
    f"{correlation_v2:.4f}"
)


# =========================================================
# SAVE RESULTS
# =========================================================

occlusion_v1.to_csv(
    "data/occlusion_v1.csv",
    index=False
)

occlusion_v2.to_csv(
    "data/occlusion_v2.csv",
    index=False
)

print("\nSaved:")
print("data/occlusion_v1.csv")
print("data/occlusion_v2.csv")