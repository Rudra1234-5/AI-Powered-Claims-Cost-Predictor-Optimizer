import pandas as pd
import streamlit as st
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from openai import AzureOpenAI
import re
from io import StringIO
import tempfile
import os
from contextlib import redirect_stdout

# Set up Streamlit page
st.set_page_config(page_title="Claims Forecasting AI", layout="wide")
st.title("🧠 AI-Powered Claims Cost Predictor")

# Azure OpenAI setup
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# Load CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Gen_AI_sample_data.csv")
        df.columns = df.columns.str.lower().str.strip()
        df["service_year_month"] = pd.to_datetime(df["service_year_month"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Sidebar
sidebar_options = ["Explore Visuals", "Ask AI for Forecast"]
sidebar_selection = st.sidebar.selectbox("Choose Option", sidebar_options)

# Forecast function using Prophet
def forecast_data(df, metric, period):
    df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=period, freq='M')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)
    st.subheader("📅 Forecasted Results")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(period))

# 📊 Static visual exploration
if sidebar_selection == "Explore Visuals":
    analysis = st.sidebar.selectbox("Select Visualization", [
        "Total Cost Over Time",
        "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost",
        "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend",
        "Top 20 Employee Costs"
    ])

    if not df.empty:
        if analysis == "Total Cost Over Time":
            grouped = df.groupby(df["service_year_month"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            grouped["service_year_month"] = grouped["service_year_month"].astype(str)
            fig = px.line(grouped, x="service_year_month", y="paid_amount", title="Total Cost Over Time")
            st.plotly_chart(fig)

        elif analysis == "Gender-wise Cost Distribution":
            grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(grouped, values="paid_amount", names="employee_gender", title="Cost by Gender")
            st.plotly_chart(fig)

        elif analysis == "Top Diagnosis by Cost":
            grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().nlargest(10).reset_index()
            fig = px.bar(grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost")
            st.plotly_chart(fig)

        elif analysis == "Average Monthly Cost Per Employee":
            df["month"] = df["service_year_month"].dt.to_period("M")
            grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            avg = grouped.groupby("month")["paid_amount"].mean().reset_index()
            avg["month"] = avg["month"].astype(str)
            fig = px.line(avg, x="month", y="paid_amount", title="Avg Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif analysis == "Diagnosis Cost Trend":
            top_diag = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            filtered = df[df["diagnosis_1_code_description"].isin(top_diag)]
            filtered["month"] = filtered["service_year_month"].dt.to_period("M")
            grouped = filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            grouped["month"] = grouped["month"].astype(str)
            fig = px.line(grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Trends")
            st.plotly_chart(fig)

        elif analysis == "Top 20 Employee Costs":
            grouped = df.groupby("employee_id")["paid_amount"].sum().nlargest(20).reset_index()
            fig = px.bar(grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Cost")
            st.plotly_chart(fig)

# 🤖 AI Forecasting with Prophet only
elif sidebar_selection == "Ask AI for Forecast":
    st.subheader("🤖 Ask Forecasting AI")
    user_input = st.text_area("What would you like to forecast?")
    
    if st.button("Ask AI") and user_input:
        try:
            # Prompt GPT to only use Prophet and be user-friendly
            prompt = f"""
You are a helpful healthcare AI.

Rules:
- ONLY use Prophet (`from prophet import Prophet`) for forecasting.
- DO NOT use or mention fprophet, ARIMA, LSTM, or any other models.
- Load data from variable `df`, with date column `service_year_month` and target `paid_amount`.
- Make the explanation user-friendly and business-oriented (not mathematical).
- Hide the code unless asked.
- Show interactive chart using `plot_plotly(model, forecast)` and display forecast using `st.dataframe()`.

Here's a preview of the dataset:
{df.head().to_string()}
"""

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            content = response.choices[0].message.content
            st.markdown("### 🧠 AI Forecast Summary")
            summary = re.sub(r"```python[\s\S]+?```", "", content)
            st.write(summary)

            # Optional: Show Python code on toggle
            if "```python" in content:
                code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    show_code = st.checkbox("Show AI-generated code")
                    if show_code:
                        st.code(code)

                    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                        tmp.write(code.encode())
                        tmp_path = tmp.name

                    output_buffer = StringIO()
                    with redirect_stdout(output_buffer):
                        exec(code, {"df": df, "st": st, "Prophet": Prophet, "plot_plotly": plot_plotly})

                    os.remove(tmp_path)

        except Exception as e:
            st.error(f"Error running AI forecast: {e}")
