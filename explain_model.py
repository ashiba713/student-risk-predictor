import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/students.csv")
X = df.drop("dropout", axis=1)
y = df["dropout"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Summary plot
shap.summary_plot(shap_values, X)
