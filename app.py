import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from streamlit_chat import message
from io import StringIO

# App title
st.set_page_config(page_title="GenAI Forecast Chatbot", layout="wide")
st.title("📊 GenAI Forecasting with Chatbot 💬")

# File upload
uploaded_file = st.file_uploader("Upload your Gen_AI.csv file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    # Basic view
    if st.checkbox("Show raw data"):
        st.write(df.head())

    # Date column selection
    date_col = st.selectbox("Select the date column", df.columns)
    value_col = st.selectbox("Select the value column (e.g., paid_amount)", df.columns)

    # Prepare data for Prophet
    forecast_df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # Forecasting horizon
    periods_input = st.slider("Forecast how many days into the future?", 30, 365, 90)

    # Run forecast
    if st.button("Run Forecast"):
        model = Prophet()
        model.fit(forecast_df)
        future = model.make_future_dataframe(periods=periods_input)
        forecast = model.predict(future)

        fig = model.plot(forecast)
        st.pyplot(fig)

        st.subheader("Forecasted Data")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# --------------------------
# Chatbot interface
# --------------------------
st.header("💬 Ask your ChatGPT-powered chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show messages
for msg in st.session_state.messages:
    message(msg["content"], is_user=msg["is_user"])

# Input
user_input = st.text_input("You:", key="input")

# Process
if user_input:
    st.session_state.messages.append({"content": user_input, "is_user": True})

    # Simulated response (You can integrate OpenAI API here)
    if "forecast" in user_input.lower():
        reply = "The forecast uses Prophet model on your uploaded data."
    elif "how many" in user_input.lower():
        reply = f"Your dataset has {len(df)} rows."
    else:
        reply = "I’m a GenAI bot! Ask me anything about your forecast or data."

    st.session_state.messages.append({"content": reply, "is_user": False})
    message(reply, is_user=False)
