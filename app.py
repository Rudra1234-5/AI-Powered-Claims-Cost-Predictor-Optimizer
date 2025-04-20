import pandas as pd
import streamlit as st
from openai import AzureOpenAI
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import re
import tempfile
import os
from contextlib import redirect_stdout
from io import StringIO

# Streamlit Title
st.title("🧠 AI-Powered Claims Forecasting & Insights")

# Azure OpenAI Client Setup (Your keys integrated)
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI_sample_data.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["service_year_month"] = pd.to_datetime(df["service_year_month"])
    return df

df = load_data()

# Sidebar Options
sidebar_options = ["Select Analysis Type", "Ask AI for Forecast"]
selection = st.sidebar.selectbox("📌 Choose Action", sidebar_options)

# Forecasting logic
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        df_prophet = df[["service_year_month", metric]].copy()
        df_prophet = df_prophet.rename(columns={"service_year_month": "ds", metric: "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
        df_prophet = df_prophet.groupby(df_prophet["ds"].dt.to_period("M").dt.to_timestamp()).sum().reset_index()

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)

        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)

        st.subheader("📈 Forecasted Data")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))

        # Show peak dynamically
        peak = forecast.loc[forecast['yhat'].idxmax()]
        st.success(f"🔍 Predicted peak: **{peak['ds'].strftime('%B %Y')}** → **{peak['yhat']:,.2f}**")

    except Exception as e:
        st.error(f"🚨 Error generating forecast: {e}")

# Section 1: Static Forecast
if selection == "Select Analysis Type":
    analysis_type = st.sidebar.selectbox("📊 Choose Analysis", [
        "Total Cost Over Time", "Gender-wise Cost", "Top Diagnosis by Cost"
    ])

    if not df.empty:
        if analysis_type == "Total Cost Over Time":
            df_grouped = df.groupby(df["service_year_month"].dt.to_period("M").dt.to_timestamp()).sum(numeric_only=True).reset_index()
            fig = px.line(df_grouped, x="service_year_month", y="paid_amount", title="Total Paid Amount Over Time")
            st.plotly_chart(fig)

        elif analysis_type == "Gender-wise Cost":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost by Gender")
            st.plotly_chart(fig)

        elif analysis_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().nlargest(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top Diagnoses")
            st.plotly_chart(fig)

# Section 2: AI Chat
elif selection == "Ask AI for Forecast":
    st.subheader("💬 Ask Healthcare AI")
    user_question = st.text_area("Type your question (e.g., What is the predicted peak month for allowed amount in 2023?)")

    if st.button("Ask AI") and user_question:
        try:
            context = f"""
You are a healthcare forecasting expert using Prophet.
Only use `Prophet` from the `prophet` library — do NOT use fbprophet or fprophet.
The dataset is already loaded as a Pandas DataFrame named `df`. Columns include service_year_month, paid_amount, allowed_amount, etc.
Ensure all forecasting uses Prophet and output is shown using plot_plotly.
Group dates using `.dt.to_period("M").dt.to_timestamp()` to avoid dtype errors.
Do NOT explain mathematically — keep responses user-friendly.
Use plotly chart for any visual output. Add a success message once output is ready.
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
            st.markdown("### 📤 AI Response")
            st.markdown(content)

            # Show "View Code" button
            if "```python" in content:
                code_block = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                if code_block:
                    python_code = code_block.group(1)
                    if st.button("👁 View Code"):
                        st.code(python_code)

                    # Execute code safely
                    try:
                        output_buffer = StringIO()
                        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                            tmp.write(python_code.encode())
                            tmp_path = tmp.name

                        with redirect_stdout(output_buffer):
                            exec(python_code, globals())

                        output = output_buffer.getvalue()
                        if output:
                            st.markdown("### 🧾 Output")
                            st.code(output)
                        else:
                            st.success("✅ Forecast completed successfully.")

                    except Exception as e:
                        st.error(f"Execution error: {e}")
                    finally:
                        os.remove(tmp_path)

        except Exception as e:
            st.error(f"AI error: {e}")
