import streamlit as st
import pandas as pd
import openai
import os
from prophet import Prophet

# Load data function (adjusted as per your columns)
@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI.csv", usecols=["service_from_date", "paid_amount", "employee_gender", "diagnosis_description", "employee_id"])
    df.columns = df.columns.str.lower().str.strip()  # Normalize column names
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    return df

# Streamlit layout
st.set_page_config(page_title="Healthcare Forecast App", layout="wide")
df = load_data()

# Sidebar options
st.sidebar.title("Healthcare Forecast & Analysis")
analysis_type = st.sidebar.selectbox("Select an analysis type:", [
    "Cost Distribution",
    "Per Employee Cost",
    "Top 5 Diagnoses",
    "Top 5 Drugs",
    "Top 5 Costliest Claims This Month",
    "Chat with AI"
])

# 'Chat with AI' option
if analysis_type == "Chat with AI":
    st.subheader("Ask Healthcare Predictions")
    ai_option = st.selectbox("Select an option:", ["Forecast Data using AI", "Custom Analysis with AI"])
    
    if ai_option == "Forecast Data using AI":
        # Forecast Data using AI option
        st.text("Provide the forecasting setup here...")
        
        # Get user input for forecasting parameters
        freq = 'Y'
        periods = st.sidebar.slider("Forecast years ahead", 1, 10, 3)
        cost_column = 'paid_amount'  # This should be the column you want to forecast
        
        # Group the data by year for forecasting
        df_grouped = df.groupby(pd.Grouper(key='service_from_date', freq=freq))[cost_column].sum().reset_index()
        df_grouped.columns = ['ds', 'y']
        df_grouped = df_grouped[df_grouped['y'] > 0].dropna()

        if len(df_grouped) > 2:
            # Prophet model for forecasting
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
        # Custom Analysis with AI option
        user_question = st.text_area("Type your custom question:")
        endpoint = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if st.button("Ask") and user_question and api_key and endpoint:
            try:
                openai.api_key = api_key
                context = f"You are a healthcare analyst. Here’s a dataset summary:\n\n{df.head().to_string()}"
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

# Rest of the analysis types (like 'Cost Distribution', 'Top 5 Diagnoses', etc.) remain unchanged
elif analysis_type == "Cost Distribution":
    group_column = st.sidebar.selectbox("Group by:", ['employee_gender', 'relationship'])
    if group_column:
        grouped = df.groupby(group_column)['paid_amount'].sum().sort_values(ascending=False)
        st.subheader(f"Cost Distribution by {group_column}")
        st.bar_chart(grouped)
        st.dataframe(grouped)

elif analysis_type == "Per Employee Cost":
    df['year'] = df['service_from_date'].dt.year
    grouped = df.groupby(['employee_id', 'year'])['paid_amount'].sum().reset_index()
    st.subheader("Per Employee Cost by Year")
    st.dataframe(grouped.head(100))

elif analysis_type == "Top 5 Diagnoses" and 'diagnosis_description' in df.columns:
    grouped = df.groupby('diagnosis_description')['paid_amount'].sum().nlargest(5)
    st.subheader("Top 5 Diagnoses by Cost")
    st.bar_chart(grouped)
    st.dataframe(grouped)

elif analysis_type == "Top 5 Drugs" and 'ndc_code' in df.columns:
    grouped = df.groupby('ndc_code')['paid_amount'].sum().nlargest(5)
    st.subheader("Top 5 Drugs by Cost")
    st.bar_chart(grouped)
    st.dataframe(grouped)

elif analysis_type == "Top 5 Costliest Claims This Month":
    latest_month = df['service_from_date'].dt.to_period('M').max()
    subset = df[df['service_from_date'].dt.to_period('M') == latest_month]
    st.subheader("Top 5 Costliest Claims This Month")
    st.dataframe(subset.nlargest(5, 'paid_amount'))

# Add the rest of the analysis types similarly as in the previous code
