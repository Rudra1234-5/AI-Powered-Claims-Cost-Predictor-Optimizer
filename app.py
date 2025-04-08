import openai
import streamlit as st
import pandas as pd
import os

# Set your OpenAI API key here
openai.api_key = "your-openai-api-key"  # Replace with your actual API key

@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI.csv")
    df.columns = df.columns.str.lower().str.strip()
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    return df

df = load_data()

st.sidebar.title("Healthcare Forecast & Analysis")
analysis_type = st.sidebar.selectbox("Select an analysis type:", [
    "Ask Healthcare Predictions"
])

if analysis_type == "Ask Healthcare Predictions":
    prediction_type = st.selectbox("Choose an AI-based prediction:", ["Forecast Data using AI", "Custom Analysis with AI"])

    if prediction_type == "Custom Analysis with AI":
        st.subheader("Ask the AI Assistant")
        user_question = st.text_area("Type your question about the data:")
        
        if st.button("Ask") and user_question:
            try:
                # Constructing the AI request
                context = f"Dataset sample:\n{df.head().to_string()}\n\nQuestion: {user_question}"
                
                # Use openai.Completion.create() for models like text-davinci or gpt-4
                response = openai.Completion.create(
                    model="gpt-4",  # You can replace this with other models like "text-davinci-003"
                    prompt=context,
                    max_tokens=150,
                    temperature=0.7
                )

                ai_response = response.choices[0].text.strip()
                st.write(ai_response)

            except Exception as e:
                st.error(f"Error with OpenAI API: {e}")
