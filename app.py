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

# 🧠 Streamlit Title
st.set_page_config(page_title="AI Claims Forecasting", layout="wide")
st.title("🧠 AI-Powered Claims Cost Predictor & Optimizer")

# 🔐 Azure OpenAI setup (with your provided key)
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# 🔄 Load data
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

# 📊 Sidebar
sidebar_options = ["Select Analysis Type", "Ask AI Forecast"]
sidebar_selection = st.sidebar.selectbox("Choose an option", sidebar_options)

# 📈 Forecast Function
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)
        st.subheader("📊 Forecasted Values")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))
    except Exception as e:
        st.error(f"Forecasting error: {e}")

# 📊 Analysis Section
if sidebar_selection == "Select Analysis Type":
    analysis_type = st.sidebar.selectbox("Choose analysis", [
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
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost by Gender")
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

# 🤖 AI Forecast Section
elif sidebar_selection == "Ask AI Forecast":
    st.subheader("🤖 Ask AI for Prophet Forecast")
    user_question = st.text_area("Ask a time-series question (e.g., 'Forecast cost next 6 months')")

    if st.button("Ask AI") and user_question:
        try:
            prompt = f"""
You are a healthcare forecasting assistant. 

📌 RULES for your response:
- You MUST USE Python's `Prophet` for all forecasting.
- DO NOT suggest or mention other models like ARIMA, LSTM, sklearn, etc.
- Data is already available in variable `df`.
- Time column is `service_year_month`, and target is `paid_amount`.
- Plot charts using `plot_plotly(model, forecast)` and show using `st.plotly_chart(fig)`.
- Show forecasted results using `st.dataframe(forecast.tail())`.
- Always wrap your code inside a Python block: ```python ... ```.

Dataset preview:
{df.head().to_string()}
"""

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            content = response.choices[0].message.content
            st.markdown("### 🧠 GPT-Generated Forecasting Code")
            st.code(content)

            # Extract and execute the code
            match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            if match:
                code = match.group(1)
                exec_globals = {
                    "pd": pd,
                    "df": df,
                    "st": st,
                    "Prophet": Prophet,
                    "plot_plotly": plot_plotly
                }

                with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                    tmp.write(code.encode())
                    tmp_path = tmp.name

                output_buffer = StringIO()
                with redirect_stdout(output_buffer):
                    exec(code, exec_globals)

                st.markdown("✅ Forecast executed successfully")
                os.remove(tmp_path)
            else:
                st.warning("⚠️ No valid Python code found in AI response.")

        except Exception as e:
            st.error(f"❌ AI error: {e}")
