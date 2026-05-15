import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -------------------
# Load model
# -------------------
model = joblib.load("models/random_forest_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

explainer = shap.TreeExplainer(model)

# -------------------
# App UI
# -------------------
st.title("Student Performance Predictor")
st.caption(
    "Machine Learning based academic risk prediction using Random Forest + feature engineering"
)

st.write(
    "Predict whether a student will pass or fail based on academic and lifestyle factors."
)

# -------------------
# User Inputs
# -------------------
age = st.number_input("Age", min_value=15, max_value=25, value=17)

studytime = st.slider("Study Time (1-4)", 1, 4, 2)

failures = st.slider("Past Class Failures", 0, 4, 0)

absences = st.number_input("Absences", min_value=0, max_value=100, value=4)

g1 = st.slider("First Period Grade (G1)", 0, 20, 10)

g2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

higher_yes = st.selectbox("Wants Higher Education?", ["Yes", "No"])

internet_yes = st.selectbox("Internet Access?", ["Yes", "No"])


# -------------------
# Prediction
# -------------------
if st.button("Predict"):

    input_dict = {col: 0 for col in feature_columns}

    # Numerical features
    input_dict["age"] = age
    input_dict["studytime"] = studytime
    input_dict["failures"] = failures
    input_dict["absences"] = absences
    input_dict["G1"] = g1
    input_dict["G2"] = g2

    # Engineered features
    input_dict["study_efficiency"] = (g1 + g2) / max(studytime, 1)
    input_dict["risk_score"] = failures * 2 + absences * 0.1
    input_dict["study_discipline"] = studytime / (absences + 1)

    # One-hot categorical features
    if "higher_yes" in input_dict:
        input_dict["higher_yes"] = 1 if higher_yes == "Yes" else 0

    if "internet_yes" in input_dict:
        input_dict["internet_yes"] = 1 if internet_yes == "Yes" else 0

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Force exact training column order
    input_df = input_df[feature_columns]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    # Result UI
    if prediction == 1:
        st.success("Prediction: PASS")

        st.metric(
            label="Pass Probability",
            value=f"{probability*100:.1f}%"
        )

        if probability > 0.8:
            st.info("Risk Level: Low")
        elif probability > 0.6:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

    else:
        st.error("Prediction: FAIL")

        st.metric(
            label="Pass Probability",
            value=f"{probability*100:.1f}%"
        )

        st.error("Risk Level: High")

    # -------------------
    # SHAP Explainability
    # -------------------
    st.markdown("---")
    st.subheader("Prediction Explanation")

    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    shap.plots.waterfall(
        shap_values[0, :, 1],
        show=False
    )

    st.pyplot(fig)