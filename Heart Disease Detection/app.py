import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Heart Disease Detection", page_icon="💓", layout="wide")
st.title("💓 Heart Disease Detection App")
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📂 Bulk Predict", "📘 Model Info"])

# ----------------------------
# TAB 1 — PREDICT
# ----------------------------
with tab1:
    st.header("Single Patient Prediction")

    # --- Input fields ---
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp_options = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    cp_label = st.selectbox("Chest Pain Type", list(cp_options.keys()))
    cp = cp_options[cp_label]

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs_label == "Yes" else 0
    restecg_options = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg_label = st.selectbox("Resting ECG Results", list(restecg_options.keys()))
    restecg = restecg_options[restecg_label]

    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang_label = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang_label == "Yes" else 0
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_label = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_options.keys()))
    slope = slope_options[slope_label]
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal_options = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
    thal_label = st.selectbox("Thalassemia", list(thal_options.keys()))
    thal = thal_options[thal_label]

    # --- Predict button ---
    if st.button("Predict Using All Models"):
        input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol,
                                fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        models = {
            "🧮 Logistic Regression": "logistic_regression_model.pkl",
            "📈 Support Vector Machine (SVM)": "svm_model.pkl",
            "🌳 Decision Tree": "decision_tree_model.pkl",
            "🌲 Random Forest": "random_forest_model.pkl"
        }

        results = {}
        for model_name, file_path in models.items():
            try:
                with open(file_path, "rb") as file:
                    model = pickle.load(file)
                pred = model.predict(input_data)[0]
                results[model_name] = "💔 Heart Disease" if pred == 1 else "💚 No Disease"
            except FileNotFoundError:
                results[model_name] = "⚠️ Model File Missing"

        st.subheader("🧠 Model Predictions")
        for model_name, result in results.items():
            if "No Disease" in result:
                st.success(f"{model_name}: {result}")
            elif "Heart Disease" in result:
                st.error(f"{model_name}: {result}")
            else:
                st.warning(f"{model_name}: {result}")


# ----------------------------
# TAB 2 — BULK PREDICT
# ----------------------------
with tab2:
    st.header("📂 Bulk Prediction (Upload CSV File)")

    # Define expected feature names (must match training)
    expected_features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

    if uploaded_file is not None:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        st.write("### 👀 Preview of Uploaded Data")
        st.dataframe(data.head())

        # Check for missing or extra columns
        missing_cols = [col for col in expected_features if col not in data.columns]
        extra_cols = [col for col in data.columns if col not in expected_features]

        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            if extra_cols:
                st.warning(f"⚠️ Ignoring extra columns: {extra_cols}")
                base_data = data[expected_features].copy()  # clean copy
            else:
                base_data = data.copy()

            # Info message
            st.info("Each model will predict independently using the same clean input features.")

            # --- Predict with all models ---
            model_files = [
                "logistic_regression_model.pkl",
                "svm_model.pkl",
                "decision_tree_model.pkl",
                "random_forest_model.pkl"
            ]

            results_summary = {}  # Store prediction counts
            final_data = base_data.copy()  # for combined output

            for model_file in model_files:
                try:
                    with open(model_file, "rb") as file:
                        model = pickle.load(file)

                    # Ensure each model gets a clean copy
                    X = base_data.copy()
                    preds = model.predict(X)

                    col_name = model_file.replace("_model.pkl", "") + "_pred"
                    final_data[col_name] = preds

                    # Count how many predicted heart disease
                    positive_cases = int(sum(preds))
                    total_cases = len(preds)
                    results_summary[col_name] = f"{positive_cases}/{total_cases} predicted Heart Disease"

                except FileNotFoundError:
                    st.warning(f"⚠️ {model_file} not found.")
                except Exception as e:
                    st.error(f"❌ Error predicting with {model_file}: {e}")

            # --- Show prediction summary ---
            st.subheader("🧠 Prediction Summary")
            if results_summary:
                summary_df = pd.DataFrame.from_dict(results_summary, orient="index", columns=["Heart Disease Count"])
                st.dataframe(summary_df)

            # --- Show results preview ---
            st.write("### 📋 Predictions from All Models")
            st.dataframe(final_data.head())

            # --- Download button ---
            csv = final_data.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Predictions as CSV",
                csv,
                "heart_disease_predictions.csv",
                "text/csv"
            )


# ----------------------------
# TAB 3 — MODEL INFO & VISUALS
# ----------------------------
with tab3:
    st.header("📊 Model Performance and Insights")

    try:
        with open("model_metrics.pkl", "rb") as f:
            model_metrics = pickle.load(f)

        metrics_df = pd.DataFrame(model_metrics).T.reset_index()
        metrics_df.columns = ["Model", "Accuracy", "F1-Score"]
        st.subheader("📈 Model Accuracy and F1 Score")
        st.dataframe(metrics_df.style.format({"Accuracy": "{:.2f}", "F1-Score": "{:.2f}"}))

        st.subheader("📊 Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Accuracy", y="Model", data=metrics_df, ax=ax)
        ax.set_xlim(0.7, 1.0)
        ax.set_title("Model Accuracy Comparison")
        st.pyplot(fig)

        try:
            with open("random_forest_model.pkl", "rb") as f:
                rf_model = pickle.load(f)
            if hasattr(rf_model, "feature_importances_"):
                st.subheader("🌲 Random Forest Feature Importance")
                feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
                importances = rf_model.feature_importances_
                imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax2)
                ax2.set_title("Feature Importance in Random Forest")
                st.pyplot(fig2)
        except FileNotFoundError:
            st.warning("⚠️ Random Forest model not found — skipping feature importance.")

    except FileNotFoundError:
        st.warning("⚠️ Model metrics file not found! Please add 'model_metrics.pkl'.")

    st.markdown("""
    ---
    ### 🧠 Model Insights
    - **Logistic Regression** → Simple, interpretable baseline.  
    - **SVM** → Strong at complex decision boundaries.  
    - **Decision Tree** → Intuitive but may overfit.  
    - **Random Forest** → Best general accuracy and interpretability.
    """)
