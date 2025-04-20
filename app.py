import pandas as pd
import streamlit as st
import plotly.express as px
from openai import AzureOpenAI
from prophet import Prophet
from prophet.plot import plot_plotly
import subprocess
import tempfile
import os
import re
from io import StringIO
from contextlib import redirect_stdout

# Streamlit App Title
st.title("AI-Powered Claims Cost Predictor & Optimizer")

# Azure OpenAI Client Setup
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI_sample_data.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["service_year_month"] = pd.to_datetime(df["service_year_month"])
    return df

df = load_data()

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# Forecast with Prophet
def forecast_with_prophet(df, metric, forecast_period):
    df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_period, freq='M')
    forecast = model.predict(future)
    return model, forecast

# Analysis Options
if sidebar_selection == "Select Analysis Type":
    analysis_type = st.sidebar.selectbox("Select Analysis", [
        "Total Cost Over Time", "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost", "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend Over Time", "Employee-wise Cost Distribution"
    ])

    if not df.empty:
        if analysis_type == "Total Cost Over Time":
            df_grouped = df.groupby(df["service_year_month"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            df_grouped["service_year_month"] = df_grouped["service_year_month"].astype(str)
            fig = px.line(df_grouped, x="service_year_month", y="paid_amount", title="Total Paid Amount Over Time")
            st.plotly_chart(fig)

        elif analysis_type == "Gender-wise Cost Distribution":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
            st.plotly_chart(fig)

        elif analysis_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().nlargest(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost")
            st.plotly_chart(fig)

        elif analysis_type == "Average Monthly Cost Per Employee":
            df["month"] = df["service_year_month"].dt.to_period("M")
            df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
            df_avg["month"] = df_avg["month"].astype(str)
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif analysis_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)]
            df_filtered["month"] = df_filtered["service_year_month"].dt.to_period("M")
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].astype(str)
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time")
            st.plotly_chart(fig)

        elif analysis_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().nlargest(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

# Forecast Chatbot Section
elif sidebar_selection == "Ask Healthcare Predictions":
    st.subheader("Ask Healthcare Predictions")
    prediction_option = st.selectbox("Choose Option", ["Forecast Data using Prophet", "Ask Forecasting Question with AI"])

    if prediction_option == "Forecast Data using Prophet":
        metric = st.selectbox("Select Metric", ["paid_amount"])
        forecast_period = st.number_input("Forecast Period (months)", min_value=1, max_value=12, value=3)

        if st.button("Generate Forecast"):
            model, forecast = forecast_with_prophet(df, metric, forecast_period)
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig)
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))

    elif prediction_option == "Ask Forecasting Question with AI":
        user_question = st.text_area("Ask a forecasting question:")
        if st.button("Ask AI") and user_question:
            try:
                context = f"""
You are an expert healthcare forecaster. You are using Python's `Prophet` library to answer all forecasting-related questions. 
You must load the data from the file: 'Gen_AI_sample_data.csv'. 
Always use `Prophet` to forecast and derive all numeric answers (e.g., peak month or predicted value).
Always show the forecast plot using `plotly` and return insights like predicted peak month or value from the `forecast` dataframe.
Here is a summary of the dataset structure:\n{df.head().to_string()}
"""

                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )

                content = response.choices[0].message.content
                st.markdown("### 💬 AI Response")
                st.code(content)

                python_code = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                if python_code:
                    code = python_code.group(1)
                    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                        tmp.write(code.encode())
                        tmp_path = tmp.name

                    output_buffer = StringIO()
                    with redirect_stdout(output_buffer):
                        exec(code, {"__name__": "__main__"})

                    output = output_buffer.getvalue()
                    st.markdown("### 🔢 Output")
                    st.code(output or "✅ Code executed successfully")

                    os.remove(tmp_path)
                else:
                    st.warning("No executable Python code found.")
            except Exception as e:
                st.error(f"Error: {e}")
