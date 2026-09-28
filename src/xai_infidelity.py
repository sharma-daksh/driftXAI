import pandas as pd
import numpy as np
import shap

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# CONFIGURATION
# =========================================================

SHAP_SAMPLE_SIZE = 300
N_PERTURBATIONS = 10
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
# 7. SELECT SAME 300 TEST SAMPLES
# =========================================================

X_eval = X2_test.iloc[
    :SHAP_SAMPLE_SIZE
].copy()


print("\n==========================================")
print("          INFIDELITY EVALUATION")
print("==========================================")

print(
    "Total Dataset 2 test samples:",
    len(X2_test)
)

print(
    "Samples used:",
    len(X_eval)
)

print(
    "Perturbations per sample:",
    N_PERTURBATIONS
)


# =========================================================
# 8. CREATE SHAP EXPLAINER
# =========================================================
#
# We explicitly ask SHAP to explain positive-class
# probability so that SHAP values and model output
# are measured in the same space.
#

background_v1 = X1_train.sample(
    n=min(100, len(X1_train)),
    random_state=RANDOM_STATE
)

background_v2 = X2_train.sample(
    n=min(100, len(X2_train)),
    random_state=RANDOM_STATE
)


explainer_v1 = shap.TreeExplainer(
    model_v1,
    background_v1,
    feature_perturbation="interventional",
    model_output="probability"
)

explainer_v2 = shap.TreeExplainer(
    model_v2,
    background_v2,
    feature_perturbation="interventional",
    model_output="probability"
)


# =========================================================
# 9. CALCULATE SHAP VALUES
# =========================================================

print("\nCalculating SHAP values...")

shap_v1 = explainer_v1.shap_values(
    X_eval
)

shap_v2 = explainer_v2.shap_values(
    X_eval
)


# =========================================================
# 10. HANDLE DIFFERENT SHAP OUTPUT FORMATS
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
# 11. MODEL OUTPUT FUNCTION
# =========================================================

def positive_probability(model, X):

    probabilities = model.predict_proba(X)

    positive_class_index = list(
        model.classes_
    ).index(1)

    return probabilities[
        :,
        positive_class_index
    ]


# =========================================================
# 12. INFIDELITY CALCULATION
# =========================================================
#
# For each original sample x:
#
#   1. Randomly perturb features.
#   2. Calculate actual model-output change.
#   3. Calculate SHAP-predicted output change.
#   4. Compare them using squared error.
#
# Infidelity:
#
# mean[
#   (actual_change - shap_change)^2
# ]
#
# Lower = better agreement.
#


def calculate_infidelity(
    model,
    shap_values,
    X,
    baseline_values,
    model_name
):

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    all_squared_errors = []

    # -----------------------------------------------------
    # Process each sample
    # -----------------------------------------------------

    for sample_index in range(len(X)):

        original = X.iloc[
            sample_index
        ]

        original_df = pd.DataFrame(
            [original],
            columns=X.columns
        )

        original_probability = (
            positive_probability(
                model,
                original_df
            )[0]
        )

        phi = shap_values[
            sample_index
        ]

        # -------------------------------------------------
        # Generate random perturbation masks
        # -------------------------------------------------
        #
        # 30% probability that a feature will be changed.
        #

        masks = rng.random(
            (
                N_PERTURBATIONS,
                len(feature_columns)
            )
        ) < 0.30

        perturbed = pd.DataFrame(
            np.tile(
                original.values,
                (
                    N_PERTURBATIONS,
                    1
                )
            ),
            columns=feature_columns
        )

        # -------------------------------------------------
        # Apply perturbations
        # -------------------------------------------------

        for feature_index, feature in enumerate(
            feature_columns
        ):

            changed_rows = masks[
                :,
                feature_index
            ]

            perturbed.loc[
                changed_rows,
                feature
            ] = baseline_values[
                feature
            ]

        # -------------------------------------------------
        # Actual model changes
        # -------------------------------------------------

        perturbed_probabilities = (
            positive_probability(
                model,
                perturbed
            )
        )

        actual_changes = (
            original_probability
            - perturbed_probabilities
        )

        # -------------------------------------------------
        # Input perturbation
        # -------------------------------------------------

        perturbation = (
            np.tile(
                original.values,
                (
                    N_PERTURBATIONS,
                    1
                )
            )
            - perturbed.values
        )

        # -------------------------------------------------
        # SHAP-predicted changes
        # -------------------------------------------------

        shap_predicted_changes = (
            perturbation @ phi
        )

        # -------------------------------------------------
        # Squared error
        # -------------------------------------------------

        squared_errors = (
            actual_changes
            - shap_predicted_changes
        ) ** 2

        all_squared_errors.extend(
            squared_errors.tolist()
        )

    infidelity = np.mean(
        all_squared_errors
    )

    print("\n==========================================")
    print(
        f"          INFIDELITY - {model_name}"
    )
    print("==========================================")

    print(
        "Infidelity:",
        round(infidelity, 6)
    )

    return infidelity


# =========================================================
# 13. BASELINE VALUES
# =========================================================

baseline_v1 = {}

baseline_v2 = {}

for feature in feature_columns:

    baseline_v1[feature] = (
        X1_train[feature]
        .mode()
        .iloc[0]
    )

    baseline_v2[feature] = (
        X2_train[feature]
        .mode()
        .iloc[0]
    )


# =========================================================
# 14. RUN MODEL V1
# =========================================================

infidelity_v1 = calculate_infidelity(
    model_v1,
    shap_v1,
    X_eval,
    baseline_v1,
    "MODEL V1"
)


# =========================================================
# 15. RUN MODEL V2
# =========================================================

infidelity_v2 = calculate_infidelity(
    model_v2,
    shap_v2,
    X_eval,
    baseline_v2,
    "MODEL V2"
)


# =========================================================
# 16. FINAL COMPARISON
# =========================================================

print("\n==========================================")
print("          INFIDELITY COMPARISON")
print("==========================================")

print(
    "Model V1:",
    round(infidelity_v1, 6)
)

print(
    "Model V2:",
    round(infidelity_v2, 6)
)


# =========================================================
# 17. SAVE RESULTS
# =========================================================

results = pd.DataFrame({
    "Model": [
        "Model V1",
        "Model V2"
    ],
    "Infidelity": [
        infidelity_v1,
        infidelity_v2
    ]
})

results.to_csv(
    "data/infidelity_results.csv",
    index=False
)

print("\nSaved:")
print("data/infidelity_results.csv")