import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
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

# Dynamically identify important columns
columns = df.columns.tolist()
date_column = 'service_from_date'
procedure_col = next((col for col in columns if 'procedure' in col), None)
drug_col = next((col for col in columns if 'ndc' in col or 'drug' in col), None)
diagnosis_col = next((col for col in columns if 'diagnosis' in col), None)
age_band_col = next((col for col in columns if 'age' in col and 'band' in col), None)

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
    "Chat with AI"
])

amount_cols = [col for col in columns if 'amount' in col]
cost_column = st.sidebar.selectbox("Select cost column:", amount_cols if amount_cols else columns)

# Analysis and charts here...

# Chatbot section - Move it to the main area
if analysis_type == "Chat with AI":
    st.title("Chat with AI Assistant")
    st.subheader("Ask questions about the healthcare data")
    
    user_question = st.text_area("Type your question here:")
    
    if user_question:
        st.markdown(f"**Your Question:** {user_question}")
    
    endpoint = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if the button is pressed and input is valid
    if st.button("Ask AI") and user_question and api_key and endpoint:
        try:
            openai.api_key = api_key
            context = f"You are a helpful analyst. Here's a healthcare dataset summary:\n\n{df.head().to_string()}"
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": user_question}
            ]
            with st.spinner('Generating response...'):
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    api_key=api_key,
                    base_url=endpoint
                )
            
            # Extract AI's response and display it
            answer = response.choices[0].message["content"]
            st.write("**AI Response:**")
            st.write(answer)  # Display the response in the app
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.write("API Response Error. Check the details above.")
