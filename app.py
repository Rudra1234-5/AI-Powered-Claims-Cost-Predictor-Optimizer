import streamlit as st
import pandas as pd
import openai
import os
from prophet import Prophet

st.set_page_config(page_title="Healthcare Cost Predictor", layout="wide")

# Caching the data loading to prevent repeated reads
@st.cache_data
def load_data():
    # Only load essential columns for analysis (you can add/remove as per your needs)
    df = pd.read_csv("Gen_AI.csv", usecols=["service_from_date", "paid_amount", "employee_gender", "diagnosis_description", "employee_id"])
    df.columns = df.columns.str.lower().str.strip()  # Normalize column names
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    return df

df = load_data()

# Dynamically identify important columns for processing
columns = df.columns.tolist()
date_column = 'service_from_date'
cost_column = 'paid_amount'

# Function to generate forecast with caching
@st.cache_data
def generate_forecast(df_grouped, periods):
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    model.fit(df_grouped)
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    return forecast

st.sidebar.title("Healthcare Analysis & Prediction Dashboard")

# Removed "Forecast" from here
analysis_type = st.sidebar.selectbox("Select Type of Analysis:", [
    "Cost Distribution",
    "Per Employee Cost",
    "Top 5 Diagnoses",
    "Top 5 Drugs",
    "Top 5 Costliest Claims This Month",
    "Inpatient vs Outpatient Costs",
    "Chronic Disease %",
    "Monthly Trends",
    "Year-over-Year Comparison",
    "Top 5 Hospitals by Spend",
    "Average Cost by Employee Age Band",
    "Total Claims by Provider Specialty",
    "Top 5 Service Types by Cost",
    "Most Common Diagnosis Categories",
    "Cost Comparison by Gender",
    "Top 5 Employers by Total Claims",
    "Trend of Cost Over Time by Relationship",
    "In-Network vs Out-of-Network Spend",
    "Claim Spend by Place of Service",
])

# Distinct option for Healthcare Prediction/Insights below the analysis types
healthcare_prediction_option = st.sidebar.button("Healthcare Predictions with AI")

if healthcare_prediction_option:
    # First Option: Choose Forecast or Custom Analysis
    st.title("AI-Powered Healthcare Predictions")
    st.subheader("Choose your desired prediction type")

    prediction_option = st.selectbox(
        "Select an option:",
        ["Forecast Healthcare Data using AI", "Custom Data Analysis with AI"]
    )

    if prediction_option == "Forecast Healthcare Data using AI":
        st.subheader("Forecast Healthcare Cost Trends")

        amount_cols = [col for col in columns if 'amount' in col]
        cost_column = st.selectbox("Select cost column for forecasting:", amount_cols if amount_cols else columns)

        # Forecast logic with data selection
        freq = 'Y'
        periods = st.sidebar.slider("Forecast years ahead", 1, 10, 3)
        df_grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[cost_column].sum().reset_index()
        df_grouped.columns = ['ds', 'y']
        df_grouped = df_grouped[df_grouped['y'] > 0].dropna()

        if len(df_grouped) > 2:
            with st.spinner('Generating forecast...'):
                forecast = generate_forecast(df_grouped, periods)

            st.subheader("Forecasted Results (Yearly)")
            fig1 = Prophet.plot(forecast)
            st.pyplot(fig1)

            forecast_df = forecast[['ds', 'yhat']]
            st.dataframe(forecast_df.tail(periods))

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
        else:
            st.warning("Not enough yearly data to forecast.")

    elif prediction_option == "Custom Data Analysis with AI":
        st.subheader("Custom Data Query Analysis")
        user_question = st.text_area("Ask your custom question about the healthcare data:")

        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("OPENAI_API_BASE")

        if not api_key or not endpoint:
            st.error("API Key or Endpoint not set. Please check your environment variables.")

        if st.button("Submit to AI") and user_question and api_key and endpoint:
            try:
                openai.api_key = api_key
                context = f"You are a healthcare analyst. Here is a summary of the dataset:\n\n{df.head().to_string()}"
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]

                with st.spinner('Processing AI response...'):
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        api_key=api_key,
                        base_url=endpoint
                    )

                if 'choices' in response and len(response['choices']) > 0:
                    answer = response['choices'][0]['message']['content']
                    st.write("**AI Response:**")
                    st.write(answer)
                else:
                    st.error("No response from AI. Please try again later.")
            
            except Exception as e:
                st.error(f"Error: {e}")
