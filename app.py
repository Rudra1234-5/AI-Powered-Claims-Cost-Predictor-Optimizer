import streamlit as st
import pandas as pd
import openai
import os

st.set_page_config(page_title="Healthcare Forecast App", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI.csv")
    df.columns = df.columns.str.lower().str.strip()  # Normalize column names
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    return df

df = load_data()

columns = df.columns.tolist()
date_column = 'service_from_date'
cost_column = 'paid_amount'  # Adjust this based on your actual dataset column

st.sidebar.title("Healthcare Forecast & Analysis")
analysis_type = st.sidebar.selectbox("Select an analysis type:", [
    "Forecast",
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
healthcare_prediction_option = st.sidebar.button("Ask Healthcare Predictions")

if healthcare_prediction_option:
    # First Option: Forecast Data
    st.title("AI Healthcare Predictions")
    st.subheader("Select an option for AI-based healthcare predictions")

    prediction_option = st.selectbox(
        "Choose an option:",
        ["Forecast Healthcare Data", "Custom Analysis with AI"]
    )

    if prediction_option == "Forecast Healthcare Data":
        # Forecasting option for healthcare data
        st.subheader("Forecast Data")
        
        # Select cost column for forecasting
        amount_cols = [col for col in columns if 'amount' in col]
        cost_column = st.selectbox("Select cost column for forecasting:", amount_cols if amount_cols else columns)

        # Add forecast logic here (same as before)
        freq = 'Y'
        periods = st.sidebar.slider("Forecast years ahead", 1, 10, 3)
        df_grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[cost_column].sum().reset_index()
        df_grouped.columns = ['ds', 'y']
        df_grouped = df_grouped[df_grouped['y'] > 0].dropna()

        if len(df_grouped) > 2:
            from prophet import Prophet
            model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
            model.fit(df_grouped)
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)

            st.subheader("Forecasted Results (Yearly)")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            forecast_df = forecast[['ds', 'yhat']]
            st.dataframe(forecast_df.tail(periods))

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
        else:
            st.warning("Not enough yearly data to forecast.")

    elif prediction_option == "Custom Analysis with AI":
        # Custom analysis with AI
        st.subheader("Custom Analysis")
        user_question = st.text_area("Type your custom question about the healthcare data:")

        # Get API key and endpoint from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("OPENAI_API_BASE")

        if not api_key or not endpoint:
            st.error("API Key or Endpoint not set. Please check your environment variables.")

        # Check if the button is pressed and input is valid
        if st.button("Ask AI") and user_question and api_key and endpoint:
            try:
                openai.api_key = api_key
                context = f"You are a helpful healthcare analyst. Here's a dataset summary:\n\n{df.head().to_string()}"
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]
                
                with st.spinner('Generating response...'):
                    # Make the API call
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        api_key=api_key,
                        base_url=endpoint
                    )

                # Check the response format and extract content
                if 'choices' in response and len(response['choices']) > 0:
                    answer = response['choices'][0]['message']['content']
                    st.write("**AI Response:**")
                    st.write(answer)  # Display the response in the app
                else:
                    st.error("No response from AI. Please try again later.")
            
            except Exception as e:
                st.error(f"Error: {e}")
