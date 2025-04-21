import pandas as pd
import streamlit as st
import plotly.express as px
from openai import AzureOpenAI
from prophet import Prophet
from prophet.plot import plot_plotly
import subprocess
import tempfile
import os
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

# Helper to format $ columns
def format_currency(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: f"${x:,.2f}")
    return df

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

# Forecasting with Prophet
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
        st.plotly_chart(fig)

        st.subheader("Forecasted Data")
        forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period)
        forecast_display = format_currency(forecast_display, ["yhat", "yhat_lower", "yhat_upper"])
        st.write(forecast_display)
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

# Load data
df = load_data()

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# Analysis
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
            df_grouped = df.groupby(df["service_year_month"].dt.to_period("M").dt.to_timestamp()).sum(numeric_only=True).reset_index()
            df_grouped["service_year_month"] = df_grouped["service_year_month"].astype(str)
            fig = px.line(df_grouped, x="service_year_month", y="paid_amount", title="Total Paid Amount ($) Over Time")
            fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
            st.plotly_chart(fig)

        elif prediction_type == "Gender-wise Cost Distribution":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
            st.plotly_chart(fig)

        elif prediction_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost ($)")
            fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat=",")
            st.plotly_chart(fig)

        elif prediction_type == "Average Monthly Cost Per Employee":
            df["month"] = df["service_year_month"].dt.to_period("M").dt.to_timestamp()
            df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
            df_avg["month"] = df_avg["month"].astype(str)
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee ($)")
            fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
            st.plotly_chart(fig)

        elif prediction_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)].copy()
            df_filtered["month"] = df_filtered["service_year_month"].dt.to_period("M").dt.to_timestamp()
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].astype(str)
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time ($)")
            fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
            st.plotly_chart(fig)

        elif prediction_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost ($)")
            fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
            st.plotly_chart(fig)

# AI-Powered Section
elif sidebar_selection == "Ask Healthcare Predictions":
    st.subheader("Ask the AI Assistant")
    user_question = st.text_area("Type your question about the data:")

    if st.button("Ask") and user_question:
        try:
            # Refined context that ensures ChatGPT uses Prophet
            context = (
                "You are a data analyst assistant. "
                "You always use the Prophet library from `prophet` for any forecasting tasks. "
                "You have access to a pandas DataFrame called `df` that contains healthcare data "
                "with columns like `service_year_month`, `allowed_amount`, and `paid_amount`. "
                "Use the Prophet model to forecast future values or analyze trends. "
                "Always provide numeric answers, never generic explanations. "
                "Use data from the 'Gen_AI_sample_data.csv'. "
                "The date column is `service_year_month`. Forecasting must be done with Prophet, "
                "and output should be clear and calculated."
            )

            # Let GPT generate Python code using Prophet
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": user_question}
            ]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0  # Ensures consistency
            )
            content = response.choices[0].message.content

            st.markdown("### 🤖 GPT Response")
            st.code(content)

            # Extract and execute Python code
            python_code = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if python_code:
                python_script = python_code.group(1).strip()

                try:
                    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                        tmp.write(python_script.encode())
                        tmp_path = tmp.name

                    output_buffer = StringIO()
                    with redirect_stdout(output_buffer):
                        exec(python_script, {"df": df})  # Pass your dataframe into the scope

                    output = output_buffer.getvalue()
                    st.markdown("### ✅ Output")
                    st.code(output or "Successfully executed.")
                except Exception as e:
                    st.error(f"Execution error: {e}")
                finally:
                    os.remove(tmp_path)
            else:
                st.info("No Python code found in GPT's response.")

        except Exception as e:
            st.error(f"Error: {e}")
