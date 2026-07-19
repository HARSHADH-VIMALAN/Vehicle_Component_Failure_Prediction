# 🚗 AI-Based Predictive Maintenance System

## 📌 Project Overview

The AI-Based Predictive Maintenance System is a machine learning application developed using **Python** and **Streamlit** to predict vehicle component failures before they occur. The system analyzes sensor and operational data, evaluates equipment health, compares multiple machine learning models, and provides intelligent maintenance recommendations to reduce downtime and improve equipment reliability.

---

## 🎯 Features

- 🚗 Vehicle Component Failure Prediction
- 📂 Dataset Upload & Exploration
- 📊 Interactive Data Analysis Dashboard
- 🤖 AI Model Training
- 📈 Machine Learning Model Comparison
- 🔍 Real-Time Failure Prediction
- 📜 Prediction History Tracking
- 🛠 Maintenance Recommendation Center
- 📄 PDF Maintenance Report Generation
- 🤖 AI Expert Maintenance Opinion (Powered by Groq)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Scikit-learn
- Joblib
- ReportLab
- Groq API

---

## 🤖 Machine Learning Models

The application compares multiple classification algorithms to identify the best-performing model.

- Random Forest Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Support Vector Machine (SVM)

The best-performing model is automatically saved for future predictions.

---

## 📊 Analysis & Visualizations

The application includes:

- Dataset Summary
- Missing Value Analysis
- Statistical Summary
- Feature Distribution
- Correlation Heatmap
- Scatter Plot Analysis
- Failure Distribution
- Component Health Dashboard
- KPI Cards
- Risk Gauge
- Feature Importance Analysis
- Confusion Matrix
- Model Performance Metrics
- Prediction Trend Analysis
- Maintenance Health Indicators

---

## 📈 Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Feature Importance

---

## 📁 Project Structure

```text
AI-Predictive-Maintenance/
│
├── app.py
├── dataset.csv
├── requirements.txt
├── README.md
├── vehicle_failure_model.pkl
└── best_vehicle_model.pkl
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-Predictive-Maintenance.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install streamlit pandas numpy plotly matplotlib scikit-learn joblib reportlab groq
```

---

## 🔑 API Configuration (Required)

This project includes an **AI Expert Opinion** module powered by the **Groq API**.

Before running the application, **replace the placeholder API key with your own Groq API key** in the source code.

Locate the following section:

```python
# Enter your Groq API Key here
GROQ_API_KEY = "YOUR_API_KEY"
```

Replace `"YOUR_API_KEY"` with your personal Groq API key.

You can obtain a free API key from the Groq Developer Console.

> **Note:** For security reasons, API keys are **not included** in this repository. Each user must provide and configure their own API key before using the AI Expert Opinion feature.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (typically `http://localhost:8501`).

---

## 📊 Workflow

1. Upload the vehicle maintenance dataset.
2. Explore and analyze the dataset.
3. Train the machine learning model.
4. Compare multiple algorithms.
5. Select the best-performing model.
6. Enter component operating parameters.
7. Predict component failure probability.
8. View maintenance recommendations.
9. Generate PDF maintenance reports.
10. Generate AI-powered expert maintenance opinions.

---

## 🚀 Future Enhancements

- Deep Learning-based failure prediction
- Live IoT sensor integration
- Predictive Remaining Useful Life (RUL) estimation
- Cloud deployment
- Email and SMS maintenance alerts
- Real-time industrial dashboard
- Multi-machine monitoring

---

## 👨‍💻 Author

**Harshadh Vimalan**

---

## ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub!
