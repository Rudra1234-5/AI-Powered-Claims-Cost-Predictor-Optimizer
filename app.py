import os
import openai
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from openai import AzureOpenAI

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# Azure OpenAI client setup (update keys appropriately if needed)
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# Load data from DBFS path (Parquet file)
@st.cache_data
def load_data():
    try:
        dbfs_path = os.path.join("/dbfs/team-2", "Enrollment.parquet")
        df = pd.read_parquet(dbfs_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df[[
            "service_from_date", "paid_amount", "employee_gender", 
            "diagnosis_1_code_description", "employee_id"
        ]]
        df["service_from_date"] = pd.to_datetime(df["service_from_date"], errors="coerce")
        return df.dropna(subset=["service_from_date"])
    except Exception as e:
        st.error(f"Error loading data from DBFS: {e}")
        return pd.DataFrame()

def generate_forecast_prompt(df, metric, forecast_period):
    prompt = f"Forecast the {metric} for the next {forecast_period} months based on the dataset provided, which includes columns like service_from_date, paid_amount, employee_gender, diagnosis_1_code_description, and employee_id."
    return prompt

def forecast_data_with_ai(df, metric, forecast_period):
    prompt = generate_forecast_prompt(df, metric, forecast_period)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for healthcare cost forecasting."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def custom_analysis_with_ai(custom_query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for healthcare analysis."},
            {"role": "user", "content": custom_query}
        ]
    )
    return response.choices[0].message.content

# Load the dataset
df = load_data()

# Streamlit Interface
st.title("AI-Powered Healthcare Predictions")

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

if sidebar_selection == "Select Analysis Type":
    prediction_type = st.sidebar.selectbox("Select an AI-powered Prediction Type", [
        "Total Cost Over Time",
        "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost",
        "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend Over Time",
        "Employee-wise Cost Distribution"
    ])

    if not df.empty:
        if prediction_type == "Total Cost Over Time":
            df_grouped = df.groupby(df["service_from_date"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            df_grouped["service_from_date"] = df_grouped["service_from_date"].astype(str)
            fig = px.line(df_grouped, x="service_from_date", y="paid_amount", title="Total Paid Amount Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Gender-wise Cost Distribution":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
            st.plotly_chart(fig)

        elif prediction_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost")
            st.plotly_chart(fig)

        elif prediction_type == "Average Monthly Cost Per Employee":
            df["month"] = df["service_from_date"].dt.to_period("M")
            df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
            df_avg["month"] = df_avg["month"].astype(str)
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif prediction_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)]
            df_filtered["month"] = df_filtered["service_from_date"].dt.to_period("M")
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].astype(str)
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

elif sidebar_selection == "Ask Healthcare Predictions":
    st.subheader("Ask Healthcare Predictions")
    prediction_option = st.selectbox("Select an AI-powered Prediction Type", ["Forecast Data using AI", "Custom Analysis with AI"])

    if prediction_option == "Forecast Data using AI":
        metric = st.selectbox("Select Metric to Forecast", ["paid_amount"])
        forecast_period = st.number_input("Forecast Period (months)", min_value=1, max_value=12, value=3)

        if not df.empty:
            st.write(df.head())
            if st.button("Generate Forecast"):
                forecast_result = forecast_data_with_ai(df, metric, forecast_period)
                st.write("AI Forecast Result:", forecast_result)

    elif prediction_option == "Custom Analysis with AI":
        user_query = st.text_area("Enter Custom Analysis Query")

        if st.button("Ask AI") and user_query:
            with st.spinner("Thinking..."):
                try:
                    analysis_result = custom_analysis_with_ai(user_query)
                    st.success("AI Response:")
                    st.write(analysis_result)
                except Exception as e:
                    st.error(f"Error: {e}")
