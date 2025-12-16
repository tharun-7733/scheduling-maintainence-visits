import gradio as gr
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load your model + scaler + feature columns
clf_model = joblib.load(os.path.join(script_dir, "best_classification_model.joblib"))
clf_scaler = joblib.load(os.path.join(script_dir, "scaler_classification.joblib"))
feature_cols = joblib.load(os.path.join(script_dir, "feature_cols.joblib"))


# Preprocessing and feature enmgineering
def build_features(age, maint_cost, downtime, maint_freq, failure_count, report_text):

    d = {}

    # Basic numeric inputs
    d["Age"] = age
    d["Maintenance_Cost"] = maint_cost
    d["Downtime"] = downtime
    d["Maintenance_Frequency"] = maint_freq
    d["Failure_Event_Count"] = failure_count

    # Text features
    text = (report_text or "").lower()
    d["Report_Length"] = len(text)
    d["Word_Count"] = len(text.split())

    d["Contain_Failure"] = int(any(w in text for w in ["failure", "shutdown", "error"]))
    d["Contain_Wear"] = int(any(w in text for w in ["wear", "degradation", "deterioration"]))
    d["Contain_Replace"] = int(any(w in text for w in ["replace", "replacement"]))

    # Interactions
    d["Cost_Time_Interaction"] = d["Maintenance_Cost"] * d["Downtime"]
    d["Severity_Score"] = d["Failure_Event_Count"] * d["Downtime"]
    d["Efficiency"] = d["Maintenance_Cost"] / (d["Downtime"] + 1)

    # Nonlinear
    d["Downtime_SQ"] = downtime**2
    d["Failure_SQ"] = failure_count**2
    d["Cost_SQ"] = maint_cost**2
    d["Age_SQ"] = age**2

    # Age group
    if age <= 5: d["Age_Group"] = 1
    elif age <= 10: d["Age_Group"] = 2
    elif age <= 20: d["Age_Group"] = 3
    elif age <= 50: d["Age_Group"] = 4
    else: d["Age_Group"] = 5

    # Text issue flags
    failure_types = {
        "battery__Issue": ["battery", "power", "drain"],
        "sensor__Issue": ["sensor", "signal", "detect"],
        "circuit__Issue": ["circuit", "board", "electric"],
        "software__Issue": ["software", "firmware", "update"],
        "overheat__Issue": ["heat", "overheat", "temp"],
        "delay__Issue": ["delay", "slow", "lag"]
    }

    for col, words in failure_types.items():
        d[col] = int(any(w in text for w in words))

    d["FailRate_Per_Age"] = failure_count / (age + 1)
    d["Downtime_Per_Event"] = downtime / (failure_count + 1)
    d["Downtime_cu"] = downtime**3
    d["Cost_cu"] = maint_cost**3
    d["Failure_cu"] = failure_count**3

    d["Failure_X_Downtime"] = d["Contain_Failure"] * downtime
    d["Sensor_X_Age"] = d["sensor__Issue"] * age
    d["Battery_X_Cost"] = d["battery__Issue"] * maint_cost

    d["Cost_Is_Outlier"] = int(maint_cost > 15000)
    d["Downtime_Is_Outlier"] = int(downtime > 40)

    d["Report_Severity_Level"] = (
        3*d["Contain_Failure"] +
        2*d["Contain_Wear"] +
        1*d["Contain_Replace"]
    )

    df_row = pd.DataFrame([d])

    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0

    return df_row[feature_cols]


def predict_failure(age, maint_cost, downtime, maint_freq, failure_count, report_text):

    row = build_features(age, maint_cost, downtime, maint_freq, failure_count, report_text)
    scaled = clf_scaler.transform(row)

    # Get probability output from Logistic Regression
    probs = clf_model.predict_proba(scaled)[0]

    p_low = probs[0]
    p_high = probs[2]

    # New "risk score"
    risk_score = 1*p_low + 3*p_high

    # Categorization
    if risk_score < 1.8:
        level = "LOW RISK"
        color = "green"
        days = 60
    elif risk_score < 2.4:
        level = "WARNING RISK"
        color = "orange"
        days = 30
    else:
        level = "HIGH RISK"
        color = "red"
        days = 7

    next_date = (datetime.now().astimezone() + timedelta(days=days))
    formatted = next_date.strftime('%Y-%m-%d %H:%M:%S %Z')

    return f"""
    <div style='font-size:20px; font-weight:bold; color:{color};'>
    Maintenance Risk: {level}<br>
    Risk Score: {risk_score:.2f}<br>
    Recommended Next Maintenance: {formatted}
    </div>
    """

with gr.Blocks() as demo:
    gr.Markdown("# 🔧 Medical Device Maintenance Prediction")
    
    with gr.Row():
        age = gr.Slider(0, 30, value=5, label="Device Age (years)")
        maint_cost = gr.Slider(0, 20000, value=3000, step=100, label="Last Maintenance Cost")

    with gr.Row():
        downtime = gr.Slider(0, 50, value=10, label="Downtime (hours)")
        maint_freq = gr.Slider(0, 20, value=2, label="Maintenance Frequency")

    failure_count = gr.Slider(0, 10, value=1, label="Failure Event Count")
    report_text = gr.Textbox(lines=3, label="Technician Notes / Issues Reported")

    btn = gr.Button("Predict Maintenance Risk")
    output = gr.HTML()

    btn.click(
        predict_failure,
        inputs=[age, maint_cost, downtime, maint_freq, failure_count, report_text],
        outputs=output
    )

demo.launch()
