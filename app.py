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
    return forecast, model

# Main application logic

# Explore Visuals Section
if sidebar_selection == "Explore Visuals":
    # Add your analysis/visualization code here (like plotting total cost, gender-wise cost, etc.)
    st.write("Explore different visualizations based on the data.")

# AI Forecasting Section
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

            # Show loading spinner while AI processes the forecast
            with st.spinner('Generating forecast...'):
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

            # Start the actual forecast generation here:
            st.subheader("🔮 Forecasting Analysis")
            forecast, model = forecast_data(df, "paid_amount", 12)  # Forecast for 12 months ahead
            st.write("Forecast generated successfully!")

            # Plot forecast
            fig = plot_plotly(model, forecast)  # Make sure to pass model and forecast to plot_plotly
            st.plotly_chart(fig)

            # Display dynamic insights: The AI will generate this based on the forecast results
            st.markdown("### 📊 Dynamic Insights:")
            st.write("🔴 AI Insights: The AI will interpret the forecast and provide dynamic insights based on your question.")
            st.write(content)

        except Exception as e:
            st.error(f"Error running AI forecast: {e}")
