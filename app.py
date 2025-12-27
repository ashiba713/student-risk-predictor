import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import sklearn
import joblib

# Load data
df = pd.read_csv("data/students.csv")
X = df.drop("dropout", axis=1)
y = df["dropout"]

# Train model
model = LogisticRegression()
model.fit(X, y)

st.title("🎓 Student Dropout & Burnout Risk Predictor")

st.write("Enter student details:")

attendance = st.slider("Attendance (%)", 0, 100, 75)
marks = st.slider("Internal Marks", 0, 100, 70)
assignments = st.slider("Assignments Completion (%)", 0, 100, 80)
sleep = st.slider("Sleep Hours (per night)", 0, 10, 6)
screen = st.slider("Screen Time (hrs/day)", 0, 12, 6)
stress = st.slider("Stress Level (1–10)", 1, 10, 5)
finance = st.selectbox("Financial Pressure", ["Low", "High"])

finance_val = 1 if finance == "High" else 0

input_data = pd.DataFrame([[attendance, marks, assignments, sleep,
                            screen, stress, finance_val]],
                          columns=X.columns)

if st.button("Predict Risk"):
    risk = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if risk == 1:
        st.error(f"⚠️ High Risk Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {prob:.2f})")

