import pandas as pd
import streamlit as st
import plotly.express as px
from openai import AzureOpenAI
from prophet import Prophet
from prophet.plot import plot_plotly
import re
from io import StringIO
import tempfile
import os
from contextlib import redirect_stdout

# 🔹 Streamlit App Title
st.set_page_config(page_title="AI Claims Forecasting", layout="wide")
st.title("🧠 AI-Powered Claims Cost Predictor & Optimizer")

# 🔐 Azure OpenAI Client Setup (your keys)
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# 🔄 Load Data
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

# 🔮 Prophet Forecasting Function
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)
        st.subheader("📈 Forecasted Data")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))
    except Exception as e:
        st.error(f"Forecast error: {e}")

# 🔀 Sidebar Options
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# 📊 Data Analysis Section
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
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
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
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

# 🤖 AI Forecast Chat Section
elif sidebar_selection == "Ask Healthcare Predictions":
    st.subheader("🤖 Ask Healthcare Forecasting Question")
    user_question = st.text_area("Ask your forecasting question (e.g., 'Predict paid_amount for next 6 months')")

    if st.button("Ask AI"):
        try:
            # GPT Prompt enforcing Prophet usage
            context = f"""
You are a forecasting expert using Python's `Prophet` for healthcare claim predictions in a Streamlit app.

Rules:
- Use only `Prophet` for time-series forecasting.
- Data is already loaded as `df`, time column is `service_year_month`, target column is `paid_amount`.
- Do not print, instead use `st.plotly_chart(fig)` for visualizations and `st.dataframe()` for results.
- Always wrap your answer in valid Python code using triple backticks.
- Plot with `plot_plotly`, not matplotlib.

Sample dataset:
{df.head().to_string()}
"""

            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": user_question}
            ]

            # Get response
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            content = response.choices[0].message.content

            # Display GPT response
            st.markdown("### 🧠 GPT-Generated Code")
            st.code(content)

            # Execute if it's valid Python code
            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if code_match:
                code = code_match.group(1)

                # Create global context
                exec_globals = {
                    "pd": pd,
                    "df": df,
                    "st": st,
                    "Prophet": Prophet,
                    "plot_plotly": plot_plotly
                }

                # Run code safely
                exec(code, exec_globals)
            else:
                st.warning("⚠️ No Python code block found.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
