# Customer Churn Prediction with Retention Cost Model

## Project Overview
This project uses machine learning to predict customer churn and identify customers who may be at risk of leaving a business. It also estimates potential profit at risk and provides a basic retention-offer recommendation.

## Objectives
- Analyze customer churn patterns.
- Train a Logistic Regression classification model.
- Estimate churn probabilities and assign risk levels.
- Estimate potential profit at risk using stated business assumptions.
- Visualize churn patterns and high-risk customers.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- CSV data processing
- Machine Learning: Logistic Regression

## Dataset
The project uses a small, synthetic dataset of 20 customers. Features include:
- Customer tenure
- Monthly charges
- Contract type
- Support calls
- Late payments
- Monthly usage
- Churn status

The data is created for demonstration and does not represent real customers.

## Project Workflow
1. Load and clean the customer dataset.
2. Prepare numerical and categorical features.
3. Split the dataset into training and test sets.
4. Train a Logistic Regression model.
5. Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.
6. Generate customer churn probabilities and risk levels.
7. Estimate potential profit at risk and compare it with a hypothetical retention-offer cost.
8. Export prediction reports and visualization charts.

## Initial Results
- Total sample customers: 20
- Customers who churned: 8
- Customers who stayed: 12
- Test-set accuracy: 83.33% (6 test records)

**Important:** The test set is extremely small, so this accuracy is only an initial demonstration result and should not be treated as evidence of real-world model performance.

## Retention Cost Assumptions
The retention analysis assumes a $10 offer cost per customer and a 40% profit margin on monthly charges. These are illustrative assumptions, not verified financial data. The estimated value is not a causal estimate of how much a retention offer would actually save.

## Visualizations
The project generates:
- `churn_distribution.png`
- `churn_risk_levels.png`
- `top_churn_risk_customers.png`

## How to Run

Install the required packages:

```bash
python -m pip install pandas scikit-learn matplotlib
```

Run the churn prediction model:

```bash
python churn_prediction.py
```

Generate the visualizations:

```bash
python visualize_churn.py
```

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   └── customer_data.csv
├── output/
│   ├── cleaned_customer_data.csv
│   ├── churn_predictions.csv
│   ├── churn_summary.csv
│   ├── churn_distribution.png
│   ├── churn_risk_levels.png
│   └── top_churn_risk_customers.png
├── churn_prediction.py
├── visualize_churn.py
├── README.md
└── .gitignore
```

## Future Improvements
- Train and evaluate models on a larger, representative dataset.
- Compare Logistic Regression with Random Forest and other classifiers.
- Use cross-validation and class-appropriate evaluation metrics.
- Validate retention assumptions using real business costs and outcomes.
- Build an interactive dashboard for business users.