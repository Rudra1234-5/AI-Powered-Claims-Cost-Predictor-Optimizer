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

# Now the chatbot is not an option in the analysis dropdown
# Add a distinct section for Chatbot in the main area
st.title("Healthcare Forecast & Analysis")

# Chatbot Section
st.header("Chat with AI Assistant")
st.subheader("Ask questions about the healthcare data")

user_question = st.text_area("Type your question here:")

# Get API key and endpoint from environment variables
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("OPENAI_API_BASE")

# Ensure API key and endpoint are set
if not api_key or not endpoint:
    st.error("API Key or Endpoint not set. Please check your environment variables.")

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
            # Make the API call
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                api_key=api_key,
                base_url=endpoint
            )

        # Print full API response for debugging
        st.write("Full API Response:")
        st.write(response)  # This will show the full response object

        # Check the response format and extract content
        if 'choices' in response and len(response['choices']) > 0:
            answer = response['choices'][0]['message']['content']
            st.write("**AI Response:**")
            st.write(answer)  # Display the response in the app
        else:
            st.error("No response from AI. Please try again later.")
        
    except Exception as e:
        st.error(f"Error: {e}")
