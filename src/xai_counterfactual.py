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
RANDOM_STATE = 42

VALID_VALUES = [-1, 0, 1]


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
    column
    for column in df1.columns
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

X1_train = X1.loc[train_indices]
y1_train = y1.loc[train_indices]

X2_train = X2.loc[train_indices]
y2_train = y2.loc[train_indices]

X2_test = X2.loc[test_indices]


# =========================================================
# 5. TRAIN MODEL V1
# =========================================================

model_v1 = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
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
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model_v2.fit(
    X2_train,
    y2_train
)


# =========================================================
# 7. XAI EVALUATION SAMPLES
# =========================================================

X_eval = X2_test.iloc[
    :SHAP_SAMPLE_SIZE
].copy()


print("\n==========================================")
print("       COUNTERFACTUAL EVALUATION")
print("==========================================")

print(
    "Dataset 2 test samples:",
    len(X2_test)
)

print(
    "Samples used:",
    len(X_eval)
)


# =========================================================
# 8. SHAP VALUES
# =========================================================

print("\nCalculating SHAP values...")

explainer_v1 = shap.TreeExplainer(
    model_v1
)

explainer_v2 = shap.TreeExplainer(
    model_v2
)

shap_v1 = explainer_v1.shap_values(
    X_eval
)

shap_v2 = explainer_v2.shap_values(
    X_eval
)


# =========================================================
# 9. HANDLE SHAP FORMAT
# =========================================================

def get_positive_class_shap(values):

    if isinstance(values, list):
        return values[1]

    if len(values.shape) == 3:
        return values[:, :, 1]

    return values


shap_v1 = get_positive_class_shap(
    shap_v1
)

shap_v2 = get_positive_class_shap(
    shap_v2
)


# =========================================================
# 10. COUNTERFACTUAL SEARCH
# =========================================================
#
# We first search for a SINGLE valid feature change
# that flips the model prediction.
#
# This is an exhaustive one-feature counterfactual
# search across all 30 features.
#
# If multiple changes work, we choose the one that
# produces the largest probability movement toward
# the opposite class.
#


def find_counterfactual(
    model,
    sample,
    original_prediction
):

    candidates = []
    original_probability = (
        model.predict_proba(sample)[0]
    )

    class_list = list(
        model.classes_
    )

    original_class_index = (
        class_list.index(
            original_prediction
        )
    )

    original_confidence = (
        original_probability[
            original_class_index
        ]
    )

    # Target = opposite class
    target_classes = [
        c for c in class_list
        if c != original_prediction
    ]

    target_class = target_classes[0]

    target_class_index = (
        class_list.index(
            target_class
        )
    )

    # -----------------------------------------------------
    # Create all one-feature candidates
    # -----------------------------------------------------

    for feature in feature_columns:

        original_value = sample.iloc[
            0
        ][feature]

        for new_value in VALID_VALUES:

            # Don't create a pointless change
            if new_value == original_value:
                continue

            modified = sample.copy()

            modified.loc[
                modified.index,
                feature
            ] = new_value

            candidates.append({
                "feature": feature,
                "old_value": original_value,
                "new_value": new_value,
                "sample": modified
            })

    # -----------------------------------------------------
    # Batch candidate predictions
    # -----------------------------------------------------

    candidate_data = pd.concat(
        [
            item["sample"]
            for item in candidates
        ],
        ignore_index=True
    )

    candidate_probabilities = (
        model.predict_proba(
            candidate_data
        )
    )

    candidate_predictions = model.predict(
        candidate_data
    )

    # -----------------------------------------------------
    # Find candidates that flip prediction
    # -----------------------------------------------------

    successful = []

    for i, candidate in enumerate(
        candidates
    ):

        if (
            candidate_predictions[i]
            == target_class
        ):

            target_probability = (
                candidate_probabilities[
                    i,
                    target_class_index
                ]
            )

            probability_change = (
                target_probability
                - (
                    1
                    - original_confidence
                )
            )

            successful.append({
                "feature": candidate["feature"],
                "old_value": candidate["old_value"],
                "new_value": candidate["new_value"],
                "target_probability": target_probability,
                "probability_change": probability_change
            })

    # -----------------------------------------------------
    # No one-feature counterfactual found
    # -----------------------------------------------------

    if not successful:

        return None

    # -----------------------------------------------------
    # Choose strongest successful counterfactual
    # -----------------------------------------------------

    successful = sorted(
        successful,
        key=lambda x: x["probability_change"],
        reverse=True
    )

    return successful[0]


# =========================================================
# 11. RUN COUNTERFACTUAL TEST
# =========================================================

def run_counterfactual_test(
    model,
    shap_values,
    model_name
):

    results = []

    for i in range(
        len(X_eval)
    ):

        sample = X_eval.iloc[
            [i]
        ]

        original_prediction = (
            model.predict(sample)[0]
        )

        original_probability = (
            model.predict_proba(sample)[0]
        )

        class_list = list(
            model.classes_
        )

        prediction_index = (
            class_list.index(
                original_prediction
            )
        )

        original_confidence = (
            original_probability[
                prediction_index
            ]
        )

        counterfactual = (
            find_counterfactual(
                model,
                sample,
                original_prediction
            )
        )

        if counterfactual is None:

            results.append({
                "Sample": i,
                "Original_Prediction":
                    original_prediction,
                "Original_Confidence":
                    original_confidence,
                "Counterfactual_Found":
                    False,
                "Feature":
                    None,
                "Old_Value":
                    None,
                "New_Value":
                    None,
                "Target_Probability":
                    None,
                "Probability_Change":
                    None,
                "SHAP_Importance":
                    None
            })

            continue

        # Find SHAP importance of counterfactual feature
        feature_index = (
            feature_columns.index(
                counterfactual["feature"]
            )
        )

        shap_importance = abs(
            shap_values[
                i,
                feature_index
            ]
        )

        results.append({
            "Sample": i,
            "Original_Prediction":
                original_prediction,
            "Original_Confidence":
                original_confidence,
            "Counterfactual_Found":
                True,
            "Feature":
                counterfactual["feature"],
            "Old_Value":
                counterfactual["old_value"],
            "New_Value":
                counterfactual["new_value"],
            "Target_Probability":
                counterfactual[
                    "target_probability"
                ],
            "Probability_Change":
                counterfactual[
                    "probability_change"
                ],
            "SHAP_Importance":
                shap_importance
        })

    results = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    success_rate = (
        results["Counterfactual_Found"]
        .mean()
    )

    successful_results = results[
        results["Counterfactual_Found"]
    ]

    if len(successful_results) > 0:

        mean_probability_change = (
            successful_results[
                "Probability_Change"
            ].mean()
        )

        correlation, p_value = (
            spearmanr(
                successful_results[
                    "SHAP_Importance"
                ],
                successful_results[
                    "Probability_Change"
                ]
            )
        )

    else:

        mean_probability_change = np.nan
        correlation = np.nan
        p_value = np.nan

    print("\n==========================================")
    print(
        f"   COUNTERFACTUAL TEST - {model_name}"
    )
    print("==========================================")

    print(
        "One-feature counterfactual success rate:",
        round(success_rate * 100, 2),
        "%"
    )

    print(
        "Successful counterfactuals:",
        len(successful_results),
        "/",
        len(results)
    )

    print(
        "Mean target-probability improvement:",
        round(
            mean_probability_change,
            4
        )
        if not np.isnan(
            mean_probability_change
        )
        else "N/A"
    )

    print(
        "SHAP vs counterfactual correlation:",
        round(correlation, 4)
        if not np.isnan(correlation)
        else "N/A"
    )

    print(
        "p-value:",
        round(p_value, 6)
        if not np.isnan(p_value)
        else "N/A"
    )

    if len(successful_results) > 0:

        print(
            "\nMost frequently selected "
            "counterfactual features:"
        )

        print(
            successful_results[
                "Feature"
            ]
            .value_counts()
            .head(10)
            .to_string()
        )

    return results


# =========================================================
# 12. RUN V1
# =========================================================

counterfactual_v1 = (
    run_counterfactual_test(
        model_v1,
        shap_v1,
        "MODEL V1"
    )
)


# =========================================================
# 13. RUN V2
# =========================================================

counterfactual_v2 = (
    run_counterfactual_test(
        model_v2,
        shap_v2,
        "MODEL V2"
    )
)


# =========================================================
# 14. SAVE RESULTS
# =========================================================

counterfactual_v1.to_csv(
    "data/counterfactual_v1.csv",
    index=False
)

counterfactual_v2.to_csv(
    "data/counterfactual_v2.csv",
    index=False
)


# =========================================================
# 15. FINAL SUMMARY
# =========================================================

summary = pd.DataFrame({
    "Model": [
        "Model V1",
        "Model V2"
    ],
    "Counterfactual_Success_Rate": [
        counterfactual_v1[
            "Counterfactual_Found"
        ].mean(),

        counterfactual_v2[
            "Counterfactual_Found"
        ].mean()
    ]
})


summary.to_csv(
    "data/counterfactual_summary.csv",
    index=False
)


print("\n==========================================")
print("      COUNTERFACTUAL RESULTS SAVED")
print("==========================================")

print(
    "data/counterfactual_v1.csv"
)

print(
    "data/counterfactual_v2.csv"
)

print(
    "data/counterfactual_summary.csv"
)