import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import openai

# Set OpenAI API keys and endpoint (with your provided API key, version, and endpoint)
openai.api_key = "8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS"
openai.api_base = "https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"  # Your Azure endpoint
openai.api_version = "2024-10-21"  # Your API version

# Streamlit page configuration
st.title("AI-Powered Claims Cost Predictor & Optimizer")

# Function to load data
def load_data():
    try:
        df = pd.read_csv("Gen_AI_sample_data.csv")  # Load your CSV file
        df.columns = df.columns.str.lower().str.strip()  # Clean column names
        df["service_year_month"] = pd.to_datetime(df["service_year_month"].astype(str))  # Ensure correct datetime format
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to forecast data using Prophet
def forecast_data_with_prophet(df, metric, forecast_period):
    try:
        # Prepare the data for Prophet
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        
        # Ensure 'ds' is in datetime format (if it's a period, convert it to timestamp)
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])  # This ensures the 'ds' column is in datetime format
        
        # Initialize the Prophet model
        model = Prophet()  # Initialize Prophet model
        model.fit(df_prophet)  # Fit the model
        
        # Make future predictions (for the specified forecast_period)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)  # Get forecast
        
        # Plot the forecast using Plotly
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig)
        
        # Display the forecast data for the future period
        st.subheader("Forecasted Data")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))
        
        # Extract peak month
        peak_month = forecast.loc[forecast['yhat'].idxmax(), ['ds', 'yhat']]
        st.write(f"Predicted Peak Month: {peak_month['ds'].strftime('%B %Y')} with an estimated value of {peak_month['yhat']:,.2f}")
        
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

# Load data
df = load_data()

# Sidebar Navigation
sidebar_options = ["Select Analysis Type", "Ask AI for Forecast"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

# Analysis Type Section
if sidebar_selection == "Select Analysis Type":
    prediction_type = st.sidebar.selectbox("Select an AI-powered Prediction Type", [
        "Total Cost Over Time",
        "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost",
        "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend Over Time",
        "Employee-wise Cost Distribution"
    ])

    if not df.empty:
        if prediction_type == "Total Cost Over Time":
            df_grouped = df.groupby(df["service_year_month"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            df_grouped["service_year_month"] = df_grouped["service_year_month"].astype(str)
            fig = px.line(df_grouped, x="service_year_month", y="paid_amount", title="Total Paid Amount Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Gender-wise Cost Distribution":
            df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
            fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
            st.plotly_chart(fig)

        elif prediction_type == "Top Diagnosis by Cost":
            df_grouped = df.groupby("diagnosis_1_code_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_1_code_description", orientation="h", title="Top 10 Diagnoses by Cost")
            st.plotly_chart(fig)

        elif prediction_type == "Average Monthly Cost Per Employee":
            df["month"] = df["service_year_month"].dt.to_period("M")
            df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
            df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
            df_avg["month"] = df_avg["month"].astype(str)
            fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
            st.plotly_chart(fig)

        elif prediction_type == "Diagnosis Cost Trend Over Time":
            top_diagnoses = df["diagnosis_1_code_description"].value_counts().nlargest(5).index
            df_filtered = df[df["diagnosis_1_code_description"].isin(top_diagnoses)]
            df_filtered["month"] = df_filtered["service_year_month"].dt.to_period("M")
            df_grouped = df_filtered.groupby(["month", "diagnosis_1_code_description"])["paid_amount"].sum().reset_index()
            df_grouped["month"] = df_grouped["month"].astype(str)
            fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_1_code_description", title="Diagnosis Cost Trend Over Time")
            st.plotly_chart(fig)

        elif prediction_type == "Employee-wise Cost Distribution":
            df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
            st.plotly_chart(fig)

# AI-Powered Forecasting Section (Optional)
elif sidebar_selection == "Ask AI for Forecast":
    metric = st.sidebar.selectbox("Select Metric for Forecast", ["paid_amount", "allowed_amount"])
    forecast_period = st.sidebar.number_input("Number of months for forecast", min_value=1, max_value=24, value=12)
    
    if st.button("Generate Forecast"):
        st.write("Generating forecast...")
        st.spinner("Processing... Please wait.")  # Spinner message
        forecast_data_with_prophet(df, metric, forecast_period)
        st.success("Forecast generation complete!")
