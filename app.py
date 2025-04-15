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
import matplotlib.pyplot as plt

st.title("AI-Powered Claims Cost Predictor & Optimizer")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# Load data
def load_data():
    try:
        df = pd.read_csv("Gen_AI_sample_data.csv")
        df.columns = df.columns.str.lower().str.strip()
        df["service_year_month"] = pd.to_datetime(df["service_year_month"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Forecasting using Prophet
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)

        # Plot (plotly)
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)

        # Optional: also show matplotlib version
        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        st.subheader("Forecasted Data")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

# Load the dataset
df = load_data()

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# Analysis Section
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
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
            st.plotly_chart(fig)

        elif prediction_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost")
            st.plotly_chart(fig)

        elif prediction_type == "Average Monthly Cost Per Employee":
            df["month"] = df["service_year_month"].dt.to_period("M")
            df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
            df_avg["month"] = df_avg["month"].astype(str)
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif prediction_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)]
            df_filtered["month"] = df_filtered["service_year_month"].dt.to_period("M")
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].astype(str)
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

# GPT Interaction Section
elif sidebar_selection == "Ask Healthcare Predictions":
    st.subheader("Ask Healthcare Predictions")
    prediction_option = st.selectbox("Select an AI-powered Prediction Type", ["Forecast Data using Prophet", "Chat with AI"])

    if prediction_option == "Forecast Data using Prophet":
        metric = st.selectbox("Select Metric to Forecast", ["paid_amount"])
        forecast_period = st.number_input("Forecast Period (months)", min_value=1, max_value=12, value=3)

        if not df.empty and st.button("Generate Forecast"):
            forecast_data_with_prophet(df, metric, int(forecast_period))

    elif prediction_option == "Chat with AI":
        st.subheader("Ask the AI Assistant")
        user_question = st.text_area("Type your question about the data:")
        if st.button("Ask") and user_question:
            try:
                context = f"""You are a helpful healthcare analyst. Here's a summary of the dataset:
{df.head().to_string()}

Columns include:
- service_year_month: date of service
- paid_amount: cost incurred
- employee_gender: gender of employee
- diagnosis_1_code_description: diagnosis info
- employee_id: employee identifier

Use `from prophet import Prophet` for forecasts. Plot with `st.pyplot()` or `st.plotly_chart()`. The dataset file is Gen_AI_sample_data.csv.
"""
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]
                response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                content = response.choices[0].message.content
                st.markdown("### 🤖 GPT Response")
                st.code(content)

                # Bash + Python detection
                bash_code = re.search(r"```bash\n(.*?)```", content, re.DOTALL)
                python_code = re.search(r"```python\n(.*?)```", content, re.DOTALL)

                if bash_code:
                    st.markdown("### 🐚 Executing Bash")
                    try:
                        bash_output = subprocess.check_output(bash_code.group(1), shell=True, stderr=subprocess.STDOUT, text=True)
                        st.code(bash_output)
                    except subprocess.CalledProcessError as e:
                        st.error(f"Bash error:\n{e.output}")
                else:
                    st.info("No Bash code detected.")

                if python_code:
                    st.markdown("### 🐍 Executing Python")
                        try:
                            python_script = python_code.group(1).strip()
                    
                            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                                tmp.write(python_script.encode())
                                tmp_path = tmp.name
                    
                            output_buffer = StringIO()
                            with redirect_stdout(output_buffer):
                                # Pass the original DataFrame into the GPT's code execution scope
                                exec(python_script, {"df": df.copy(), "pd": pd, "st": st, "px": px})
                    
                            output = output_buffer.getvalue()
                            st.code(output or "✅ Executed successfully")
                    
                        except Exception as e:
                            st.error(f"Python error: {str(e)}")
                    
                        finally:
                            os.remove(tmp_path)
