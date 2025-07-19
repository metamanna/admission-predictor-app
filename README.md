# admission-predictor-app
A Streamlit web application that predicts a student's chance of university admission based on academic scores, university rating, and research experience using machine learning models like Linear Regression, Random Forest, and XGBoost. Includes interactive visualizations, explainable AI (SHAP), and model performance metrics.

# 🎓 Student Admission Predictor

A web-based ML application built using **Streamlit** to predict the **probability of student admission** to a university based on standardized test scores, academic performance, and research experience.

This project demonstrates end-to-end machine learning integration with model comparison, visualization, and explainability using **SHAP**.

---

## 📌 Key Features

- ✅ Predict admission chance based on academic inputs
- 🔄 Model comparison: Linear Regression, Random Forest, XGBoost
- 📉 Display of performance metrics (R², MAE, RMSE)
- 🧠 Explainable AI: SHAP-based visual explanations for predictions
- 💻 Clean, interactive UI using Streamlit
- 💾 Model persistence with `joblib`
- 🚀 Deployable on Streamlit Cloud

---

## 📂 Dataset

- **Source**: Kaggle - [Graduate Admission Dataset](https://www.kaggle.com/mohansacharya/graduate-admissions)
- **Target Variable**: `Chance of Admit`
- **Features Used**:
  - GRE Score
  - TOEFL Score
  - University Rating
  - SOP (Statement of Purpose)
  - LOR (Letter of Recommendation)
  - CGPA (Undergraduate GPA)
  - Research (0 = No, 1 = Yes)

---

## 🔧 Tech Stack

- Python 🐍
- Streamlit 📺
- Pandas, NumPy, Scikit-learn 📊
- XGBoost, SHAP, joblib

---

## 🚀 How to Run Locally

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/student-admission-predictor.git
cd student-admission-predictor

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run aasignment7_admissionpredictor.py

📈 Model Evaluation
Model	R² Score	MAE	RMSE
Linear Regression	0.821	0.048	0.068
Random Forest	0.807	0.050	0.071
XGBoost	0.766	0.056	0.078

🧠 SHAP Explainability
The app uses SHAP (SHapley Additive exPlanations) to visualize how each input feature affects the prediction.
Force Plot
Bar Plot
Waterfall Plot
These explainability features help in understanding model decisions.

📤 Deployment
Deploy the app for free using Streamlit Cloud:
streamlit run aasignment7_admissionpredictor.py

📤 Deployment
Deploy the app for free using Streamlit Cloud:
streamlit run aasignment7_admissionpredictor.py

Streamlit app link: https://admission-predictor-app-gasu4evpvaojj26f2gvqpn.streamlit.app/
