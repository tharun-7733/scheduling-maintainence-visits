🏥 Predictive Maintenance for Medical Equipment
Machine Learning • Real‑Time Failure Risk Prediction
<p align="center"> <img src="https://cdn-icons-png.flaticon.com/512/2966/2966489.png" width="120"/> </p> <p align="center"> <img src="https://img.shields.io/badge/Project%20Type-Machine%20Learning-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/Domain-Healthcare%20Tech-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/Status-Active-orange?style=for-the-badge"/> </p>
🔍 Overview
This project predicts the maintenance risk level of biomedical equipment by analyzing:

🧮 Numerical data (device age, downtime, cost, failure history)

📝 Text logs written by technicians

🤖 Machine Learning models such as Logistic Regression, Random Forest, XGBoost

The goal is to shift hospitals from reactive → proactive → predictive maintenance.

✨ Features
Feature	Description
⚙️ ML-based risk prediction	Predicts if a device is High or Low risk
📝 NLP on maintenance logs	Extracts meaning from real technician notes
📊 Interactive Dashboard	Real-time prediction interface with Gradio
🚀 Scalable Pipeline	Works with any hospital dataset
💾 Model + Scaler Saving	Ensures accurate prediction consistency
🧠 Project Architecture

(If you want, I can replace this with a custom PNG made from your flowchart.)

📁 Folder Structure
├── data/
│   └── Medical_Device_Failure_dataset.csv
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_list.pkl
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── app/
│   └── app.py
└── README.md
🚀 How to Run
🔧 Install Requirements
pip install -r requirements.txt
🏋️ Train the Model
python src/train_model.py
🌐 Launch Web UI
python app/app.py
Your browser will open the prediction interface automatically ✔️

📊 Model Performance

Logistic Regression chosen as final model

Best generalization & lowest overfitting

Good performance on both structured + text features

🧩 Workflow Diagram
flowchart TD
    A[Device Usage Data] --> C[Feature Engineering]
    B[Maintenance Log Text] --> D[NLP Processing]
    C --> E[Merge Features]
    D --> E
    E --> F[Train Models<br>LogReg | RF | XGBoost]
    F --> G[Evaluate<br>AUC • F1 • Accuracy]
    G --> H{Select Best Model}
    H --> I[Save Model + Scaler + Features]
    I --> J[Real-Time Prediction UI]
🔮 Future Scope
IoT sensor integration (real-time streaming)

Deep learning for text understanding (BERT, RoBERTa)

Cloud-based maintenance dashboard

Remaining Useful Life (RUL) prediction

Multi-hospital centralized management

