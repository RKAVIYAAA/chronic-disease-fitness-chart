import gradio as gr
import pandas as pd
import joblib
import torch
from transformers import pipeline

# Load your trained model (ensure model.pkl is accessible locally)
model = joblib.load("model.pkl")

# Specify exact pre-trained HF model for sentiment-analysis pipeline
nlp_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Mapping dictionaries
gender_map = {"Male": 0, "Female": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}
yes_no_map = {"No": 0, "Yes": 1}

def predict(age, gender, bmi, blood_pressure, cholesterol, glucose,
            activity, smoking, alcohol, family_history, user_text):
    # Prepare input data for your ML model
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

    # Predict chronic disease risk
    prediction = model.predict(input_df)[0]
    risk_label = "High Risk ⚠️" if prediction == 1 else "Low Risk ✅"

    # Analyze sentiment of user input text with specified pipeline
    nlp_result = nlp_pipeline(user_text)[0]
    nlp_summary = f"Sentiment: {nlp_result['label']} (confidence {nlp_result['score']:.2f})"

    return risk_label, nlp_summary

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 100, value=30, label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(value=27.0, label="BMI"),
        gr.Number(value=120, label="Blood Pressure (mmHg)"),
        gr.Number(value=200, label="Cholesterol (mg/dL)"),
        gr.Number(value=100, label="Glucose (mg/dL)"),
        gr.Radio(["Low", "Moderate", "High"], label="Physical Activity"),
        gr.Radio(["No", "Yes"], label="Smoking Status"),
        gr.Radio(["No", "Yes"], label="Alcohol Intake"),
        gr.Radio(["No", "Yes"], label="Family History of Chronic Disease"),
        gr.Textbox(lines=2, placeholder="Enter text for sentiment analysis", label="Input Text (Optional)")
    ],
    outputs=["text", "text"],
    title="Chronic Disease Fitness Chart with Sentiment Analysis",
    description="Predict your chronic disease risk and analyze text sentiment."
)

if __name__ == "__main__":
    iface.launch()


