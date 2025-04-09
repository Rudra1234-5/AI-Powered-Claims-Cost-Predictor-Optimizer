import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import datetime

# Load OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set page title
st.set_page_config(page_title="AI-Powered Healthcare Predictor", layout="wide")
st.title("AI-Powered Healthcare Predictor")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI.csv", usecols=[
        "service_from_date", "paid_amount", "employee_gender",
        "diagnosis_description", "employee_id"])
    df["service_from_date"] = pd.to_datetime(df["service_from_date"])
    return df

df = load_data()

# Sidebar options
st.sidebar.title("Select an option")
option = st.sidebar.radio("", [
    "Forecasting Analysis",
    "Cost Trend Dashboard",
    "Ask Healthcare Predictions"])

# Forecasting
if option == "Forecasting Analysis":
    st.subheader("Forecasting Analysis")
    gender = st.selectbox("Select Gender", df["employee_gender"].unique())
    diag = st.selectbox("Select Diagnosis", df["diagnosis_description"].unique())

    filtered = df[(df["employee_gender"] == gender) & (df["diagnosis_description"] == diag)]
    daily = filtered.groupby("service_from_date")["paid_amount"].sum().reset_index()

    fig = px.line(daily, x="service_from_date", y="paid_amount",
                  title=f"Daily Paid Amount for {diag} ({gender})")
    st.plotly_chart(fig, use_container_width=True)

# Dashboard
elif option == "Cost Trend Dashboard":
    st.subheader("Cost Trend Dashboard")
    st.plotly_chart(px.histogram(df, x="diagnosis_description", y="paid_amount",
                                 color="employee_gender", barmode="group",
                                 title="Paid Amount by Diagnosis and Gender"), use_container_width=True)

# AI Predictions
elif option == "Ask Healthcare Predictions":
    st.subheader("AI-Powered Healthcare Predictions")
    prediction_type = st.radio("Select an AI-powered Prediction Type", [
        "Forecast Data using AI", "Custom Analysis with AI"])

    if prediction_type == "Forecast Data using AI":
        st.info("Enter details to generate a forecast prediction using GPT-4")
        prompt = st.text_area("Describe what you want to forecast:",
                              "Forecast healthcare costs for female employees with diabetes")
        if st.button("Generate Forecast"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a healthcare data analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success(response.choices[0].message.content)

    elif prediction_type == "Custom Analysis with AI":
        st.info("Enter your custom analysis query to get insights from GPT-4")
        custom_query = st.text_area("Enter Custom Analysis Query")
        if st.button("Analyze with AI"):
            with st.spinner("Analyzing with GPT-4..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a healthcare analytics assistant."},
                        {"role": "user", "content": custom_query}
                    ]
                )
                st.success(response.choices[0].message.content)
