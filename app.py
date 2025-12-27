import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/students.csv")
X = df.drop("dropout", axis=1)
y = df["dropout"]

# Train model
model = LogisticRegression()
model.fit(X, y)

st.set_page_config(page_title="Student Risk Predictor", layout="wide", page_icon="🎓")
st.title("🎓 Student Dropout & Burnout Risk Predictor")

st.write("Fill in the student details below:")

# Layout inputs in two columns
col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("📘 Attendance (%)", 0, 100, 75)
    marks = st.slider("📝 Internal Marks", 0, 100, 70)
    assignments = st.slider("📂 Assignments Completion (%)", 0, 100, 80)
    sleep = st.slider("😴 Sleep Hours (per night)", 0, 10, 6)

with col2:
    screen = st.slider("📱 Screen Time (hrs/day)", 0, 12, 6)
    stress = st.slider("⚡ Stress Level (1–10)", 1, 10, 5)
    finance = st.selectbox("💰 Financial Pressure", ["Low", "High"])

finance_val = 1 if finance == "High" else 0

input_data = pd.DataFrame([[attendance, marks, assignments, sleep,
                            screen, stress, finance_val]],
                          columns=X.columns)

if st.button("🚀 Predict Risk"):
    risk = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if risk == 1:
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background:linear-gradient(90deg,#ff4d4d,#ff9999); color:white;">
                ⚠️ <b>High Risk Detected</b><br>
                Probability: {prob:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(prob * 100))
    else:
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background:linear-gradient(90deg,#00c851,#33b5e5); color:white;">
                ✅ <b>Low Risk</b><br>
                Probability: {prob:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(prob * 100))

    # Show feature importance chart
    st.subheader("📊 Feature Importance (Logistic Regression Coefficients)")
    importance = pd.Series(model.coef_[0], index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=importance, y=importance.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    st.pyplot(fig)