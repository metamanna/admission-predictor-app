import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Admission_Predict.csv")

df.drop(columns=["Serial No."], inplace=True)

# Show the first few rows
# df.head()

# df.isnull().sum()

plt.figure(figsize=(15, 8))
df.hist(bins=20, figsize=(15, 10), color='skyblue')
plt.suptitle("Feature Distributions", fontsize=16)
# plt.show()

plt.figure(figsize=(9, 3))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
# plt.show()

# print(df.columns.tolist())

df.columns = df.columns.str.strip()

X = df.drop("Chance of Admit", axis=1)
y = df["Chance of Admit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

# print("LR R² Score:", r2_score(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Chance of Admit")
plt.ylabel("Predicted Chance of Admit")
plt.title("Actual vs Predicted (Linear Regression)")
plt.grid(True)
# plt.show()

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

import joblib
import os

# 🔁 Save or load the Random Forest model
if not os.path.exists("rf_model.pkl"):
    joblib.dump(rf_model, "rf_model.pkl")
else:
    rf_model = joblib.load("rf_model.pkl")

rf_pred = rf_model.predict(X_test)

# print(" Random Forest R²:", r2_score(y_test, rf_pred))
# print(" MAE:", mean_absolute_error(y_test, rf_pred))
# print(" RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

# 📊 Show model performance in sidebar
with st.sidebar:
    st.subheader("📊 Model Performance (Test Set)")
    st.metric("Linear Regression R²", round(r2_score(y_test, y_pred), 3))
    st.metric("Random Forest R²", round(r2_score(y_test, rf_pred), 3))
    st.metric("XGBoost R²", round(r2_score(y_test, xgb_pred), 3))

# print(" XGBoost R²:", r2_score(y_test, xgb_pred))
# print(" MAE:", mean_absolute_error(y_test, xgb_pred))
# print(" RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))

# rf_model = RandomForestRegressor()
# rf_model.fit(X_train, y_train)

explainer = shap.Explainer(rf_model, X_train)

# Example: for first test instance
# sample = X_test.iloc[[0]]  # Keep it a DataFrame
# prediction = rf_model.predict(sample)[0]

# shap_values = explainer(sample)

# # Force plot (optional for web)
# shap.plots.force(shap_values[0])

# # Waterfall plot (great for regression!)
# shap.plots.waterfall(shap_values[0])

# Show top 3 features impacting the prediction
# shap_df = pd.DataFrame({
#     'Feature': sample.columns,
#     'SHAP Value': shap_values.values[0],
#     'Value': sample.values[0]
# }).sort_values(by='SHAP Value', key=abs, ascending=False)

# top_features = shap_df.head(3)
# print(top_features)

# # Display the waterfall plot using matplotlib
# shap.plots.waterfall(shap_values[0])
# plt.title("SHAP Waterfall Plot for Sample Prediction")
# plt.show()

# Load your trained model and data
# model = joblib.load("rf_model.pkl")  # Replace with your actual model path
# X_train = pd.read_csv("X_train.csv")  # Your training features used for SHAP
model = rf_model # Using the existing trained model
# X_train is already available

feature_names = X_train.columns.tolist()

# ------------------------- SHAP Explanation Function ------------------------- #
def explain_prediction(model, X_train, sample):
    """
    Generate SHAP explanation for a single prediction.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(sample)
    prediction = model.predict(sample)[0]

    shap_df = pd.DataFrame({
        'Feature': sample.columns,
        'SHAP Value': shap_values.values[0],
        'Value': sample.values[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    top_features_df = shap_df.head(3)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()

    return prediction, top_features_df, fig

# ------------------------- Streamlit UI ------------------------- #
st.title("🎓 Student Admission Chance Predictor")

# Collect user input
user_input = {}
for feature in feature_names:
    # Set appropriate max values based on features, assuming GRE and TOEFL are out of 340 and 120 respectively, CGPA out of 10
    max_value = 100.0 # Default max value
    if feature == "GRE Score":
        max_value = 340.0
    elif feature == "TOEFL Score":
        max_value = 120.0
    elif feature == "CGPA":
        max_value = 10.0
    elif feature in ["University Rating", "SOP", "LOR", "Research"]:
         max_value = 5.0 # Assuming these are on a scale of 1-5 or similar

    user_input[feature] = st.number_input(f"Enter {feature}:", min_value=0.0, max_value=max_value)


# Convert input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Predict Admission Chance
if st.button("Predict Admission Chance"):
    prediction = model.predict(user_input_df)[0]
    st.success(f"🎯 **Predicted Chance of Admission:** `{round(prediction * 100, 2)}%`")

    # SHAP Explanation
    with st.expander("💬 Why this prediction?"):
        st.subheader("🔍 SHAP Explanation")

        pred, top_feats, shap_plot = explain_prediction(model, X_train, user_input_df)

        st.write("💡 **Top 3 Influencing Factors:**")
        for i, row in top_feats.iterrows():
            arrow = "🔼" if row['SHAP Value'] > 0 else "🔽"
            st.write(f"- {arrow} **{row['Feature']} = {row['Value']}** (Impact: `{round(row['SHAP Value'], 3)}`)")

        st.pyplot(shap_plot)
