import openai
import pandas as pd
import streamlit as st

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key_here'

@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_csv("Gen_AI.csv")
    
    # Normalize column names (strip leading/trailing spaces, lowercase all letters)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check available columns in the dataset
    st.write(f"Available columns in the dataset: {df.columns.tolist()}")
    
    # Define the columns to use for the analysis
    required_columns = [
        "service_from_date", "paid_amount", "employee_gender", 
        "diagnosis_1_code_description", "employee_id"
    ]
    
    # Check if the required columns exist in the dataset
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return None
    
    # Filter only the necessary columns
    df = df[required_columns]
    
    # Convert service_from_date to datetime (handling errors)
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    
    return df

def generate_forecast_prompt(df, metric, forecast_period):
    # Generate a prompt for the AI model to forecast the paid amount
    prompt = f"Forecast the {metric} for the next {forecast_period} months based on the dataset provided, which includes the following columns: service_from_date, paid_amount, employee_gender, diagnosis_1_code_description, and employee_id."
    return prompt

def forecast_data_with_ai(df, metric, forecast_period):
    # Generate the prompt
    prompt = generate_forecast_prompt(df, metric, forecast_period)
    
    # Call the OpenAI API to get the forecast
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=150
    )
    
    # Get the forecast result from the response
    forecast_result = response.choices[0].text.strip()
    return forecast_result

# Streamlit Interface
st.title("AI-Powered Healthcare Predictions")

# Sidebar for navigation
sidebar_options = ["Select Analysis Type", "Ask Healthcare Predictions"]
sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

if sidebar_selection == "Select Analysis Type":
    st.write("Here, users can explore data analysis options and visualization tools.")
    # Your existing analysis tools and plots code here

elif sidebar_selection == "Ask Healthcare Predictions":
    prediction_option = st.selectbox("Select an AI-powered Prediction Type", ["Forecast Data using AI", "Custom Analysis with AI"])
    
    if prediction_option == "Forecast Data using AI":
        st.subheader("Forecast Data using AI")
        # Allow the user to select the metric (e.g., paid_amount)
        metric = st.selectbox("Select Metric to Forecast", ["paid_amount"])
        
        # Allow the user to select the forecast period (e.g., 1 month, 3 months)
        forecast_period = st.number_input("Forecast Period (months)", min_value=1, max_value=12, value=3)
        
        # Get the data and forecast the selected metric
        df = load_data()
        
        if df is not None:
            # Display a preview of the data
            st.write(df.head())
            
            # Trigger AI forecast when the button is clicked
            if st.button("Generate Forecast"):
                forecast_result = forecast_data_with_ai(df, metric, forecast_period)
                st.write("AI Forecast Result: ", forecast_result)

    elif prediction_option == "Custom Analysis with AI":
        st.subheader("Custom Analysis with AI")
        # Let the user input custom query
        custom_query = st.text_area("Enter Custom Analysis Query")
        
        if st.button("Analyze with AI"):
            if custom_query:
                response = openai.Completion.create(
                    model="gpt-4",
                    prompt=custom_query,
                    max_tokens=150
                )
                analysis_result = response.choices[0].text.strip()
                st.write("AI Custom Analysis Result: ", analysis_result)
            else:
                st.error("Please enter a custom query for analysis.")
