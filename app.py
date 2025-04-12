import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO
import openai

st.set_page_config(page_title="Healthcare Forecast App", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI.csv")
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    return df

df = load_data()

date_column = 'service_from_date'
procedure_col = next((col for col in df.columns if 'procedure' in col.lower()), None)
drug_col = next((col for col in df.columns if 'ndc' in col.lower()), None)
diagnosis_col = next((col for col in df.columns if 'diagnosis' in col.lower()), None)
age_band_col = next((col for col in df.columns if 'age' in col.lower() and 'band' in col.lower()), None)

st.sidebar.title("Healthcare Forecast & Analysis")
analysis_type = st.sidebar.selectbox("Select an analysis type:", [
    "Chat with Healthcare Bot",
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

cost_column = st.sidebar.selectbox("Select cost column:", [col for col in df.columns if 'amount' in col.lower()])

if analysis_type == "Chat with Healthcare Bot":
    st.subheader("Ask Me Anything About Your Healthcare Data ✨")

    user_input = st.chat_input("Ask about forecast, cost trends, diagnoses, etc.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        columns_summary = ", ".join(df.columns[:15])
        prompt = f"""You are a healthcare data analyst bot. You are analyzing this dataset with columns: {columns_summary}. \
        The user asked: {user_input}\nAnswer with the current data insights and yearly forecast in a helpful and concise way."""

        openai.api_key = st.secrets["OPENAI_API_KEY"]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
        )

        answer = response['choices'][0]['message']['content']

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))

elif analysis_type == "Forecast":
    freq = 'Y'  # Force yearly forecast
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

# The rest of the app remains unchanged...
