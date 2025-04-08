import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO
import openai
import os

st.set_page_config(page_title="Healthcare Forecast App", layout="wide")

# Load data and ensure the columns match the dataset
@st.cache_data
def load_data():
    try:
        # Load specific columns based on the dataset structure
        df = pd.read_csv("Gen_AI.csv", usecols=[
            "service_from_date", "paid_amount", "employee_gender", "diagnosis_1_code", "employee_id"
        ])
        df.columns = df.columns.str.lower().str.strip()  # Normalize column names
        df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# Dynamically identify important columns based on the dataset
columns = df.columns.tolist()
date_column = 'service_from_date'
amount_cols = [col for col in columns if 'amount' in col]
cost_column = 'paid_amount'  # Adjusted to the correct cost column

# Sidebar settings for analysis options
st.sidebar.title("Healthcare Forecast & Analysis")
analysis_type = st.sidebar.selectbox("Select an analysis type:", [
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
    "Ask Healthcare Predictions"
])

# Check if the user selects 'Ask Healthcare Predictions'
if analysis_type == "Ask Healthcare Predictions":
    st.subheader("Choose AI Option")
    ai_option = st.radio("Select an AI-powered option:", ["Forecast Data Using AI", "Custom Analysis with AI"])
    
    if ai_option == "Forecast Data Using AI":
        # Forecast option as previously implemented
        freq = 'Y'
        periods = st.sidebar.slider("Forecast years ahead", 1, 10, 3)
        df_grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[cost_column].sum().reset_index()
        df_grouped.columns = ['ds', 'y']
        df_grouped = df_grouped[df_grouped['y'] > 0].dropna()

        if len(df_grouped) > 2:
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
    
    elif ai_option == "Custom Analysis with AI":
        # Handle custom analysis via AI (e.g., chatbot functionality)
        st.subheader("Ask the AI Assistant")
        user_question = st.text_area("Type your question about the data:")
        endpoint = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if st.button("Ask") and user_question and api_key and endpoint:
            try:
                openai.api_key = api_key
                context = f"You are a helpful analyst. Here's a healthcare dataset summary:\n\n{df.head().to_string()}"
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    api_key=api_key,
                    base_url=endpoint
                )
                st.write(response.choices[0].message["content"])
            except Exception as e:
                st.error(f"Error: {e}")

elif analysis_type == "Cost Distribution":
    group_column = st.sidebar.selectbox("Group by:", [col for col in ['employee_gender', 'relationship'] if col in columns])
    if group_column:
        grouped = df.groupby(group_column)[cost_column].sum().sort_values(ascending=False)
        st.subheader(f"Cost Distribution by {group_column}")
        st.bar_chart(grouped)
        st.dataframe(grouped)

# Other analyses can be added in the same manner for the other analysis options.
