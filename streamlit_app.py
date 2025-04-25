import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page Config
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Load Data, Model, Encoders
@st.cache_data
def load_data():
    df = pd.read_csv("processed_balanced_data.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop any "Unnamed" columns
    return df

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

df = load_data()
model = load_model()
le_dict = load_encoders()

# App Title
st.title("🔍 Credit Risk Prediction")
st.markdown("Predict whether a loan applicant is **Good** or **Bad** credit risk using a trained Random Forest model.")

# Navigation Tabs
tabs = st.tabs(["📊 Data Overview", "🔎 Model Insights", "🧠 Predict"])

# Data Overview
with tabs[0]:
    st.header("📊 Data Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # st.subheader("Credit Risk Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    sns.countplot(x='Credit_Risk', data=df, palette="cool", ax=ax1)
    ax1.set_title("Credit Risk Class Distribution")
    # st.pyplot(fig1)

    # st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='mako', ax=ax2)
    # st.pyplot(fig2)

# Model Insights
with tabs[1]:
    st.header("🔎 Model Performance")

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png")

    st.subheader("Top 10 Important Features")
    st.image("feature_importance.png")

    st.markdown("""
    ### 🔑 Insights:
    - Duration, Credit Amount, and Age are key predictors.
    - Model has strong classification capabilities.
    """)

# Prediction
with tabs[2]:
    st.header("🧠 Predict Credit Risk")
    st.markdown("### 🧾 Fill Applicant Info")

    input_data = {}
    for col in df.columns[:-1]:
        if col in le_dict:
            options = list(le_dict[col].classes_)
            selected = st.selectbox(col, options)
            input_data[col] = le_dict[col].transform([selected])[0]
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][prediction]
        label = "Good" if prediction == 0 else "Bad"

        st.success(f"💡 Credit Risk: **{label}**")
        st.info(f"📈 Confidence: `{prob:.2f}`")

        st.markdown("### 📥 Download Prediction")
        report_df = pd.DataFrame(input_data, index=[0])
        report_df['Predicted_Risk'] = label
        st.download_button("Download CSV", report_df.to_csv(index=False), "credit_risk_prediction.csv", "text/csv")

