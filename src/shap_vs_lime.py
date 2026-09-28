import pandas as pd
import numpy as np
import shap

from scipy.io import arff
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from lime.lime_tabular import LimeTabularExplainer


# =========================================================
# CONFIGURATION
# =========================================================

EVAL_SAMPLE_SIZE = 100
LIME_NUM_SAMPLES = 300
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
    column
    for column in df1.columns
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
# 7. SAME EVALUATION SAMPLES
# =========================================================

X_eval = X2_test.iloc[
    :EVAL_SAMPLE_SIZE
].copy()


print("\n==========================================")
print("          SHAP vs LIME")
print("==========================================")

print(
    "Dataset 2 test samples:",
    len(X2_test)
)

print(
    "Samples used:",
    len(X_eval)
)

print(
    "LIME perturbations per explanation:",
    LIME_NUM_SAMPLES
)


# =========================================================
# 8. SHAP
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
# 9. HANDLE SHAP OUTPUT
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
# 10. LIME CONFIGURATION
# =========================================================

categorical_features = list(
    range(len(feature_columns))
)

categorical_names = {
    index: ["-1", "0", "1"]
    for index in categorical_features
}


# LIME needs a prediction function.
# Convert its generated categorical values
# back to valid integer feature values.

def make_lime_predict_function(model):

    def predict(data):

        data = np.asarray(data)

        data = np.rint(data).astype(int)

        data = np.clip(
            data,
            -1,
            1
        )

        dataframe = pd.DataFrame(
            data,
            columns=feature_columns
        )

        return model.predict_proba(
            dataframe
        )

    return predict


lime_predict_v1 = make_lime_predict_function(
    model_v1
)

lime_predict_v2 = make_lime_predict_function(
    model_v2
)


# =========================================================
# 11. CREATE LIME EXPLAINERS
# =========================================================

lime_explainer_v1 = LimeTabularExplainer(
    training_data=X1_train.values,
    feature_names=feature_columns,
    class_names=[
        "Phishing",
        "Legitimate"
    ],
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    mode="classification",
    discretize_continuous=False,
    random_state=RANDOM_STATE
)


lime_explainer_v2 = LimeTabularExplainer(
    training_data=X2_train.values,
    feature_names=feature_columns,
    class_names=[
        "Phishing",
        "Legitimate"
    ],
    categorical_features=categorical_features,
    categorical_names=categorical_names,
    mode="classification",
    discretize_continuous=False,
    random_state=RANDOM_STATE
)


# =========================================================
# 12. COLLECT LIME IMPORTANCE
# =========================================================

def calculate_lime_importance(
    explainer,
    predict_function,
    X_data,
    model,
    model_name
):

    importance_matrix = np.zeros(
        (
            len(X_data),
            len(feature_columns)
        )
    )

    for i in range(len(X_data)):

        if (i + 1) % 10 == 0:
            print(
                f"{model_name}: "
                f"LIME explanation "
                f"{i + 1}/{len(X_data)}"
            )

        sample = X_data.iloc[i].values

        explanation = explainer.explain_instance(
            sample,
            predict_function,
            labels=[0, 1],
            num_features=len(feature_columns),
            num_samples=LIME_NUM_SAMPLES
        )

        # Determine the class predicted by the model.
        prediction = model.predict(
            X_data.iloc[[i]]
        )[0]

        # Model classes are [-1, 1].
        # Convert to index used by LIME:
        #
        # -1 -> index 0
        #  1 -> index 1

        class_index = (
            0
            if prediction == -1
            else 1
        )

        local_map = explanation.as_map()

        if class_index not in local_map:
            continue

        for feature_index, weight in local_map[
            class_index
        ]:

            importance_matrix[
                i,
                feature_index
            ] = abs(weight)

    return importance_matrix


# =========================================================
# 13. CALCULATE LIME
# =========================================================

print("\nCalculating LIME for Model V1...")

lime_v1 = calculate_lime_importance(
    lime_explainer_v1,
    lime_predict_v1,
    X_eval,
    model_v1,
    "Model V1"
)


print("\nCalculating LIME for Model V2...")

lime_v2 = calculate_lime_importance(
    lime_explainer_v2,
    lime_predict_v2,
    X_eval,
    model_v2,
    "Model V2"
)


# =========================================================
# 14. GLOBAL IMPORTANCE
# =========================================================

shap_importance_v1 = np.abs(
    shap_v1
).mean(axis=0)

shap_importance_v2 = np.abs(
    shap_v2
).mean(axis=0)

lime_importance_v1 = lime_v1.mean(
    axis=0
)

lime_importance_v2 = lime_v2.mean(
    axis=0
)


# =========================================================
# 15. COMPARE ONE MODEL
# =========================================================

def compare_explainers(
    shap_importance,
    lime_importance,
    model_name
):

    correlation, p_value = spearmanr(
        shap_importance,
        lime_importance
    )

    shap_top = set(
        np.argsort(
            shap_importance
        )[-TOP_K:]
    )

    lime_top = set(
        np.argsort(
            lime_importance
        )[-TOP_K:]
    )

    overlap = len(
        shap_top & lime_top
    )

    results = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP_Importance": shap_importance,
        "LIME_Importance": lime_importance
    })

    results = results.sort_values(
        "SHAP_Importance",
        ascending=False
    )

    print("\n==========================================")
    print(
        f"       SHAP vs LIME - {model_name}"
    )
    print("==========================================")

    print(
        results.head(10).to_string(
            index=False
        )
    )

    print(
        "\nSpearman correlation:",
        round(correlation, 4)
    )

    print(
        "p-value:",
        round(p_value, 6)
    )

    print(
        f"Top-{TOP_K} overlap:",
        f"{overlap}/{TOP_K}"
    )

    return (
        results,
        correlation,
        overlap
    )


# =========================================================
# 16. COMPARE V1
# =========================================================

comparison_v1, correlation_v1, overlap_v1 = (
    compare_explainers(
        shap_importance_v1,
        lime_importance_v1,
        "MODEL V1"
    )
)


# =========================================================
# 17. COMPARE V2
# =========================================================

comparison_v2, correlation_v2, overlap_v2 = (
    compare_explainers(
        shap_importance_v2,
        lime_importance_v2,
        "MODEL V2"
    )
)


# =========================================================
# 18. FINAL SUMMARY
# =========================================================

print("\n==========================================")
print("       EXPLAINER CONSISTENCY")
print("==========================================")

print(
    "\n                 Model V1       Model V2"
)

print(
    "SHAP-LIME correlation:",
    f"{correlation_v1:.4f}",
    "       ",
    f"{correlation_v2:.4f}"
)

print(
    f"Top-{TOP_K} overlap:",
    f"{overlap_v1}/{TOP_K}",
    "              ",
    f"{overlap_v2}/{TOP_K}"
)


# =========================================================
# 19. SAVE RESULTS
# =========================================================

comparison_v1.to_csv(
    "data/shap_lime_v1.csv",
    index=False
)

comparison_v2.to_csv(
    "data/shap_lime_v2.csv",
    index=False
)

summary = pd.DataFrame({
    "Model": [
        "Model V1",
        "Model V2"
    ],
    "SHAP_LIME_Spearman": [
        correlation_v1,
        correlation_v2
    ],
    "Top5_Overlap": [
        overlap_v1,
        overlap_v2
    ]
})

summary.to_csv(
    "data/shap_lime_summary.csv",
    index=False
)


print("\nSaved:")
print("data/shap_lime_v1.csv")
print("data/shap_lime_v2.csv")
print("data/shap_lime_summary.csv")
