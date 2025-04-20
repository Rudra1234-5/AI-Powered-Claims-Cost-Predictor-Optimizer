import pandas as pd
import streamlit as st
import plotly.express as px
from openai import AzureOpenAI
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import subprocess
import tempfile
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
import re

st.title("AI-Powered Claims Cost Predictor & Optimizer")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

def load_data():
    try:
        df = pd.read_csv("Gen_AI_sample_data.csv")
        df.columns = df.columns.str.lower().str.strip()
        df["service_year_month"] = pd.to_datetime(df["service_year_month"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to forecast data using Prophet
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)
        st.subheader("Forecasted Data")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

# Load data
df = load_data()

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# Analysis Type Section
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
            df_grouped = df.groupby(df["service_year_month"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            df_grouped["service_year_month"] = df_grouped["service_year_month"].astype(str)
            fig = px.line(df_grouped, x="service_year_month", y="paid_amount", title="Total Paid Amount Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Gender-wise Cost Distribution":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_group_
