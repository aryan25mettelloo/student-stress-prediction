import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Create Dataset
# -----------------------------
data = {
    'sleep_hours': [6, 7, 5, 8, 4, 6, 7, 5, 8, 6],
    'study_hours': [5, 6, 7, 4, 8, 5, 6, 7, 3, 6],
    'screen_time': [4, 3, 5, 2, 6, 4, 3, 5, 2, 4],
    'physical_activity': [1, 2, 0, 3, 0, 1, 2, 0, 3, 1],
    'yoga_time': [0, 10, 0, 20, 0, 10, 15, 0, 25, 10],
    'stress_level': [2, 1, 2, 0, 2, 1, 1, 2, 0, 1]
}

df = pd.DataFrame(data)

# -----------------------------
# STEP 2: Train Model
# -----------------------------
X = df.drop('stress_level', axis=1)
y = df['stress_level']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 Student Stress Prediction App")
st.write("Predict stress based on lifestyle + Yoga (IKS)")

# Inputs
sleep = st.slider("Sleep Hours", 0, 10, 6)
study = st.slider("Study Hours", 0, 10, 5)
screen = st.slider("Screen Time", 0, 10, 4)
activity = st.slider("Physical Activity (days/week)", 0, 7, 1)
yoga = st.slider("Yoga Time (minutes/day)", 0, 30, 10)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Stress Level"):
    input_data = [[sleep, study, screen, activity, yoga]]
    prediction = model.predict(input_data)

    levels = ["LOW", "MEDIUM", "HIGH"]

    st.success(f"Predicted Stress Level: {levels[prediction[0]]}")

    # Advice (IKS integration)
    if prediction[0] == 2:
        st.warning("⚠️ High stress! Try yoga & reduce screen time.")
    elif prediction[0] == 1:
        st.info("🙂 Moderate stress. Maintain balance.")
    else:
        st.success("😌 Low stress. Keep it up!")

# -----------------------------
# GRAPH (BONUS MARKS)
# -----------------------------
st.subheader("📊 Yoga vs Stress Visualization")

fig, ax = plt.subplots()
ax.scatter(df['yoga_time'], df['stress_level'])
ax.set_xlabel("Yoga Time")
ax.set_ylabel("Stress Level")
ax.set_title("Yoga vs Stress")

st.pyplot(fig)