
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load and clean data
data = pd.read_csv(DATA_DIR / "customer_data.csv")
data["contract_type"] = data["contract_type"].str.strip().str.title()
data = data.dropna().drop_duplicates()

print("\nCUSTOMER CHURN PREDICTION")
print("Customers loaded:", len(data))
print("Customers churned:", int(data["churn"].sum()))
print("Customers retained:", int((data["churn"] == 0).sum()))

# 2. Define model inputs and target
X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]

categorical_features = ["contract_type"]
numerical_features = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "late_payments",
    "monthly_usage_gb",
]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", StandardScaler(), numerical_features),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nMODEL EVALUATION")
print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
all_features = data.drop(columns=["customer_id", "churn"])
data["churn_probability"] = model.predict_proba(all_features)[:, 1]
data["predicted_churn"] = model.predict(all_features)

data["risk_level"] = data["churn_probability"].apply(
    lambda probability:
        "High" if probability >= 0.70
        else "Medium" if probability >= 0.40
        else "Low"
)
retention_offer_cost = 10.0
estimated_monthly_profit = (
    data["monthly_charges"] * 0.40
)  

data["retention_offer_cost"] = retention_offer_cost
data["estimated_monthly_profit"] = estimated_monthly_profit
data["estimated_monthly_profit_at_risk"] = (
    data["churn_probability"] * estimated_monthly_profit
)
data["net_expected_value_of_offer"] = (
    data["estimated_monthly_profit_at_risk"] - retention_offer_cost
)

data["retention_recommendation"] = data.apply(
    lambda row:
        "Consider retention offer"
        if row["net_expected_value_of_offer"] > 0
        else "Review before offering discount",
    axis=1,
)
report_path = OUTPUT_DIR / "churn_predictions.csv"
data.sort_values("churn_probability", ascending=False).to_csv(
    report_path, index=False
)

summary = pd.DataFrame({
    "metric": [
        "Total customers",
        "Actual churned customers",
        "Customers predicted high risk",
        "Average predicted churn probability (%)",
        "Retention offer cost per customer",
    ],
    "value": [
        len(data),
        int(data["churn"].sum()),
        int((data["risk_level"] == "High").sum()),
        round(data["churn_probability"].mean() * 100, 2),
        retention_offer_cost,
    ],
})
summary.to_csv(OUTPUT_DIR / "churn_summary.csv", index=False)

print("\nCHURN and RETENTION SUMMARY")
print(summary.to_string(index=False))
print("\nHighest-risk customers:")
print(
    data[["customer_id", "churn_probability", "risk_level",
          "net_expected_value_of_offer"]]
    .sort_values("churn_probability", ascending=False)
    .head(5)
    .to_string(index=False)
)
print("\nPrediction report saved to:", report_path)
print("Summary saved to:", OUTPUT_DIR / "churn_summary.csv")
