import openai
import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key_here'

client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",  
    api_version="2024-10-21",
    azure_endpoint = "https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
    )

@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_parquet("Gen_AI.csv")
    
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
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for healthcare cost forecasting."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Get the forecast result from the response
    forecast_result = response['choices'][0]['message']['content']
    return forecast_result

def custom_analysis_with_ai(custom_query):
    # Call the OpenAI API to get the custom analysis
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Replace with your model dpeloyment name.
        messages=[
            {"role": "system", "content": "You are a helpful assistant for healthcare analysis."},
            {"role": "user", "content": custom_query}
        ]
    )
    
    # Get the analysis result from the response
    analysis_result = response.choices[0].message.content
    return analysis_result
