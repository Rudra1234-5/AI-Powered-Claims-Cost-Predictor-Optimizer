import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import openai
import os

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Gen_AI.csv", usecols=[
        "service_from_date",
        "paid_amount",
        "employee_gender",
        "diagnosis_description",
        "employee_id"
    ])

df = load_data()
df["service_from_date"] = pd.to_datetime(df["service_from_date"], errors='coerce')
df.dropna(subset=["service_from_date"], inplace=True)

# Sidebar
st.sidebar.title("Select an option")
option = st.sidebar.radio("", ["AI-Powered Healthcare Predictions", "Ask Healthcare Predictions"])

if option == "AI-Powered Healthcare Predictions":
    st.title("AI-Powered Healthcare Predictions")
    prediction_type = st.sidebar.selectbox("Select an AI-powered Prediction Type", [
        "Total Cost Over Time",
        "Gender-wise Cost Distribution",
        "Top Diagnosis by Cost",
        "Average Monthly Cost Per Employee",
        "Diagnosis Cost Trend Over Time",
        "Employee-wise Cost Distribution"
    ])

    if prediction_type == "Total Cost Over Time":
        df_grouped = df.groupby(df["service_from_date"].dt.to_period("M")).sum(numeric_only=True).reset_index()
        df_grouped["service_from_date"] = df_grouped["service_from_date"].astype(str)
        fig = px.line(df_grouped, x="service_from_date", y="paid_amount", title="Total Paid Amount Over Time")
        st.plotly_chart(fig)

    elif prediction_type == "Gender-wise Cost Distribution":
        df_grouped = df.groupby("employee_gender")["paid_amount"].sum().reset_index()
        fig = px.pie(df_grouped, values="paid_amount", names="employee_gender", title="Cost Distribution by Gender")
        st.plotly_chart(fig)

    elif prediction_type == "Top Diagnosis by Cost":
        df_grouped = df.groupby("diagnosis_description")["paid_amount"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(df_grouped, x="paid_amount", y="diagnosis_description", orientation="h", title="Top 10 Diagnoses by Cost")
        st.plotly_chart(fig)

    elif prediction_type == "Average Monthly Cost Per Employee":
        df["month"] = df["service_from_date"].dt.to_period("M")
        df_grouped = df.groupby(["month", "employee_id"])["paid_amount"].sum().reset_index()
        df_avg = df_grouped.groupby("month")["paid_amount"].mean().reset_index()
        df_avg["month"] = df_avg["month"].astype(str)
        fig = px.line(df_avg, x="month", y="paid_amount", title="Average Monthly Cost Per Employee")
        st.plotly_chart(fig)

    elif prediction_type == "Diagnosis Cost Trend Over Time":
        top_diagnoses = df["diagnosis_description"].value_counts().nlargest(5).index
        df_filtered = df[df["diagnosis_description"].isin(top_diagnoses)]
        df_filtered["month"] = df_filtered["service_from_date"].dt.to_period("M")
        df_grouped = df_filtered.groupby(["month", "diagnosis_description"])["paid_amount"].sum().reset_index()
        df_grouped["month"] = df_grouped["month"].astype(str)
        fig = px.line(df_grouped, x="month", y="paid_amount", color="diagnosis_description", title="Diagnosis Cost Trend Over Time")
        st.plotly_chart(fig)

    elif prediction_type == "Employee-wise Cost Distribution":
        df_grouped = df.groupby("employee_id")["paid_amount"].sum().sort_values(ascending=False).head(20).reset_index()
        fig = px.bar(df_grouped, x="employee_id", y="paid_amount", title="Top 20 Employees by Total Cost")
        st.plotly_chart(fig)

elif option == "Ask Healthcare Predictions":
    st.title("Ask Healthcare Predictions")
    sub_option = st.radio("Select Mode", ["Forecast Data using AI", "Custom Analysis with AI"])

    if sub_option == "Forecast Data using AI":
        st.info("Coming soon: Intelligent forecasting based on your dataset.")

    elif sub_option == "Custom Analysis with AI":
        st.subheader("Custom Analysis with AI")
        user_query = st.text_area("Enter Custom Analysis Query")

        if st.button("Ask AI") and user_query:
            with st.spinner("Thinking..."):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a healthcare data expert."},
                            {"role": "user", "content": user_query}
                        ]
                    )
                    st.success("AI Response:")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error: {e}")
