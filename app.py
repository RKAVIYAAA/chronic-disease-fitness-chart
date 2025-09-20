# app1.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\R.KAVIYA\OneDrive\Desktop\gen ai\model.pkl")

model = load_model()

# -------------------------------
# App Title
# -------------------------------
st.title("🩺 Chronic Disease Fitness Chart")
st.markdown("This app predicts your **chronic disease risk** and shows your fitness position compared to population data.")

# -------------------------------
# User Inputs (Sidebar)
# -------------------------------
st.sidebar.header("Enter Your Health Details:")

age = st.sidebar.slider("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=27.0, step=0.1)
blood_pressure = st.sidebar.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=70, max_value=300, value=100)
activity = st.sidebar.selectbox("Physical Activity", ["Low", "Moderate", "High"])
smoking = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
alcohol = st.sidebar.selectbox("Alcohol Intake", ["No", "Yes"])
family_history = st.sidebar.selectbox("Family History of Chronic Disease", ["No", "Yes"])

# Encode Inputs
activity_map = {"Low": 0, "Moderate": 1, "High": 2}
yes_no_map = {"No": 0, "Yes": 1}
gender_map = {"Male": 0, "Female": 1}

input_data = {
    "age": age,
    "gender": gender_map[gender],
    "bmi": bmi,
    "blood_pressure": blood_pressure,
    "cholesterol_level": cholesterol,
    "glucose_level": glucose,
    "physical_activity": activity_map[activity],
    "smoking_status": yes_no_map[smoking],
    "alcohol_intake": yes_no_map[alcohol],
    "family_history": yes_no_map[family_history],
}

input_df = pd.DataFrame([input_data])

# -------------------------------
# Predict Button
# -------------------------------
predict_clicked = st.sidebar.button("🔍 Predict Risk")

if predict_clicked:
    # Prediction
    prediction = model.predict(input_df)[0]
    risk_label = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"

    st.subheader("Predicted Chronic Disease Risk")
    st.write(f"Based on your inputs: **{risk_label}**")

    # -------------------------------
    # Visualization: Age vs BMI
    # -------------------------------
    st.subheader("📊 Your Position on Age vs BMI")

    sample_data = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45, 50, 60, 70],
        "BMI": [18, 22, 27, 30, 25, 28, 32, 35, 29]
    })

    fig, ax = plt.subplots()
    sns.scatterplot(data=sample_data, x="Age", y="BMI", color="blue", label="Population", ax=ax)
    sns.scatterplot(x=[age], y=[bmi], color="red", s=200, label="You", ax=ax)

    ax.axhspan(18.5, 24.9, color="green", alpha=0.2, label="Healthy BMI")
    ax.axhspan(25, 29.9, color="yellow", alpha=0.2, label="Overweight")
    ax.axhspan(30, 50, color="red", alpha=0.1, label="Obese")

    ax.set_title("Age vs BMI Chart")
    ax.set_xlim(18, 100)
    ax.set_ylim(10, 50)
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Fitness Recommendations
    # -------------------------------
    st.subheader("💡 Personalized Fitness Tips")

    if bmi > 30:
        st.write("🏃 You are in the **Obese** category. Consider weight management strategies like daily walking and a balanced diet.")
    elif bmi > 25:
        st.write("⚖️ You are **Overweight**. Regular exercise and reducing processed food can help lower risk.")
    else:
        st.write("✅ Your BMI is in the healthy range. Maintain your lifestyle with regular activity.")

    if cholesterol > 240:
        st.write("🥗 Your cholesterol is high. Reduce saturated fats and increase fiber intake.")
    if glucose > 126:
        st.write("🍎 Your glucose level is high. Monitor sugar intake and consult a doctor for diabetes screening.")
    if smoking == "Yes":
        st.write("🚭 Quitting smoking will significantly reduce chronic disease risks.")
    if alcohol == "Yes":
        st.write("🍷 Reduce alcohol intake to improve heart and liver health.")
    if activity == "Low":
        st.write("🏋️ Increase your physical activity (at least 30 min of brisk walking daily).")

    st.success("🎯 Stay consistent with healthy habits for long-term fitness!")

else:
    st.info("👈 Enter your details in the sidebar and click **Predict Risk** to see results.")
