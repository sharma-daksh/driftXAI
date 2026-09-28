import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATASET
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
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop("Result", axis=1)
y = df["Result"]


# ==========================================
# 4. SAME TRAIN/TEST SPLIT AS MODEL V1
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


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
# 6. CREATE SHAP EXPLAINER
# ==========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# ==========================================
# 7. GLOBAL SHAP IMPORTANCE
# ==========================================

print("\n========== SHAP FEATURE IMPORTANCE ==========")

# Handle SHAP output format for binary classification
if isinstance(shap_values, list):
    shap_values_for_analysis = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_for_analysis = shap_values[:, :, 1]
else:
    shap_values_for_analysis = shap_values


importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": abs(shap_values_for_analysis).mean(axis=0)
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(10))


# ==========================================
# 8. SHAP SUMMARY PLOT
# ==========================================

shap.summary_plot(
    shap_values_for_analysis,
    X_test,
    show=False
)

# ==========================================
# 10. LOCAL SHAP EXPLANATION
# ==========================================

sample_index = 0

print("\n========== LOCAL EXPLANATION ==========")

sample = X_test.iloc[[sample_index]]

prediction = model.predict(sample)[0]

if prediction == -1:
    prediction_label = "PHISHING"
else:
    prediction_label = "LEGITIMATE"

print("Sample index:", sample_index)
print("Actual:", "PHISHING" if y_test.iloc[sample_index] == -1 else "LEGITIMATE")
print("Predicted:", prediction_label)


# Get SHAP values for this sample
sample_shap = shap_values_for_analysis[sample_index]

local_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Feature_Value": sample.iloc[0].values,
    "SHAP_Value": sample_shap
})

local_importance["Absolute_SHAP"] = abs(
    local_importance["SHAP_Value"]
)

local_importance = local_importance.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

print("\nTop features influencing this prediction:")

print(
    local_importance[
        ["Feature", "Feature_Value", "SHAP_Value"]
    ].head(10)
)
plt.tight_layout()
plt.show()