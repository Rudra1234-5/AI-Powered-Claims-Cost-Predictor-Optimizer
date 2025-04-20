import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from openai import AzureOpenAI
import re
import tempfile
import os
from io import StringIO
from contextlib import redirect_stdout

# --- Set up Streamlit layout ---
st.set_page_config(layout="wide")
st.title("🧠 AI-Powered Claims Cost Predictor & Optimizer")

# --- Azure OpenAI Configuration ---
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# --- Load and prepare data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI_sample_data.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["service_year_month"] = pd.to_datetime(df["service_year_month"])
    return df

df = load_data()

# --- Sidebar Navigation ---
sidebar_selection = st.sidebar.radio("📍 Choose Section", ["Select Analysis Type", "Ask AI for Forecast"])

# --- Section: Data Analysis ---
if sidebar_selection == "Select Analysis Type":
    st.header("📊 Claims Data Analysis")
    analysis_type = st.sidebar.selectbox("Select Analysis", [
        "Total Cost Over Time",
        "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost",
        "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend Over Time",
        "Employee-wise Cost Distribution"
    ])

    if not df.empty:
        if analysis_type == "Total Cost Over Time":
            df_grouped = df.groupby(df["service_year_month"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            df_grouped["service_year_month"] = df_grouped["service_year_month"].dt.to_timestamp()
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
            df_avg["month"] = df_avg["month"].dt.to_timestamp()
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif analysis_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)]
            df_filtered["month"] = df_filtered["service_year_month"].dt.to_period("M")
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].dt.to_timestamp()
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time")
            st.plotly_chart(fig)

        elif analysis_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().nlargest(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

# --- Section: Chat with AI ---
elif sidebar_selection == "Ask AI for Forecast":
    st.header("💬 AI Forecast Assistant")
    user_question = st.text_area("Ask a forecasting question related to the data 👇", height=100)

    if st.button("🔍 Ask AI"):
        with st.spinner("Generating prediction..."):
            try:
                context = f"""
You are a healthcare forecasting expert. Always use Prophet for predictions based on the user's CSV data.
The file name is: 'Gen_AI_sample_data.csv'. Use the column 'service_year_month' for time and either 'allowed_amount' or 'paid_amount' for values.
Always use `plot_plotly(model, forecast)` to display forecasts.
Avoid math-heavy or hardcoded responses — keep it friendly and dynamic.
Here is a sample of the dataset:
{df.head(2).to_string()}
"""

                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )

                ai_reply = response.choices[0].message.content
                st.markdown("### 🤖 AI Response")
                st.write(ai_reply)

                # Check for Python code block
                code_match = re.search(r"```python\n(.*?)```", ai_reply, re.DOTALL)
                if code_match:
                    python_code = code_match.group(1)

                    with st.expander("👁️ View Generated Code"):
                        st.code(python_code)

                    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
                        tmp_file.write(python_code.encode())
                        temp_file_path = tmp_file.name

                    output_buffer = StringIO()
                    try:
                        with redirect_stdout(output_buffer):
                            exec(python_code, globals())
                        output_result = output_buffer.getvalue()
                        if output_result:
                            st.text_area("📤 Code Output", output_result, height=200)
                        else:
                            st.success("✅ Forecast generated successfully.")
                    except Exception as exec_error:
                        st.error(f"Execution error: {exec_error}")
                    finally:
                        os.remove(temp_file_path)
                else:
                    st.info("⚠️ No code was detected in AI response.")

            except Exception as api_error:
                st.error(f"Error communicating with OpenAI: {api_error}")
