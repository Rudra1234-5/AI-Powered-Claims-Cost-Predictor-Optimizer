import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from prophet import Prophet
import matplotlib.pyplot as plt
import re
import tempfile
import os
from io import StringIO
from contextlib import redirect_stdout

# Initialize Azure OpenAI client with your keys
client = AzureOpenAI(
    api_key="8B86xeO8aV6pSZ9W3OqjihyeStsSxe06UIY0ku0RsPivUBIhvISnJQQJ99BDACHYHv6XJ3w3AAAAACOGf8nS",
    api_version="2024-10-21",
    azure_endpoint="https://globa-m99lmcki-eastus2.cognitiveservices.azure.com/"
)

# Sample Data (replace this with your actual data)
df = pd.read_csv('Gen_AI_sample_data.csv')

# Sidebar options
sidebar_selection = st.sidebar.radio("Select a task", ["Ask AI for Forecast", "Explore Data", "Visualization", "Statistics", "Trends"])

def forecast_with_prophet(df, metric):
    """
    Generate forecast using the Prophet model.
    """
    try:
        # Preparing data for Prophet
        df_prophet = df[["service_year_month", metric]].rename(columns={"service_year_month": "ds", metric: "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

        model = Prophet()
        model.fit(df_prophet)
        
        # Make future predictions
        future = model.make_future_dataframe(df_prophet, periods=12, freq='M')
        forecast = model.predict(future)

        # Find the peak month
        peak_month = forecast.loc[forecast['yhat'].idxmax(), ['ds', 'yhat']]
        
        # Plotting forecast
        fig = model.plot(forecast)
        
        return peak_month, fig, forecast
    except Exception as e:
        st.error(f"Error running Prophet forecast: {e}")
        return None, None, None


# Chat with AI Section
if sidebar_selection == "Ask AI for Forecast":
    st.subheader("Ask AI for Forecast")

    prediction_option = st.selectbox("Select an AI-powered Prediction Type", ["Ask about Data Trends", "Show Forecast", "Chat with AI"])

    if prediction_option == "Chat with AI":
        st.subheader("Ask the AI Assistant")

        # Collect user query
        user_question = st.text_area("Type your question about the data:")
        
        if st.button("Ask") and user_question:
            try:
                # Include dataset preview in context for OpenAI
                context = f"You are a helpful healthcare analyst. Here's a dataset summary:\n\n{df.head().to_string()}. When asked about data trends or forecasts, use the Prophet model for predictions."

                # Ask OpenAI with context and user question
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_question}
                ]
                
                # Query OpenAI model
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # or use another model if required
                    messages=messages
                )

                # Display response from AI
                st.write(response.choices[0].message.content)
                
                # Process response to check for Python code or bash script
                content = response.choices[0].message.content
                
                # Match Python or Bash code in response
                python_code = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                
                if python_code:
                    python_script = python_code.group(1).strip()
                    st.markdown("### 🐍 Python Code:")
                    with st.expander("Show Python Code"):
                        st.code(python_script)

                    # Run Python Code upon request
                    if st.button("Run Python Code"):
                        try:
                            # Using a temporary file to execute Python code
                            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                                tmp.write(python_script.encode())
                                tmp_path = tmp.name

                            output_buffer = StringIO()
                            with redirect_stdout(output_buffer):
                                exec(python_script)

                            output = output_buffer.getvalue()
                            st.code(output or "✅ Code executed successfully")
                        except Exception as e:
                            st.error(f"Python error: {e}")
                        finally:
                            os.remove(tmp_path)
                else:
                    st.info("No Python code detected.")
                
                # Check if user is asking for forecasting
                if "forecast" in user_question.lower():
                    peak_month, fig, forecast = forecast_with_prophet(df, "allowed_amount")
                    
                    if peak_month is not None:
                        st.subheader(f"Predicted Peak Month: {peak_month['ds'].strftime('%B %Y')}")
                        st.write(f"Predicted Peak Value: {peak_month['yhat']:.2f}")
                        st.plotly_chart(fig)
                        st.write(forecast.tail())  # Show last few forecasted values

            except Exception as e:
                st.error(f"Error: {e}")
    
# Explore Data Section
elif sidebar_selection == "Explore Data":
    st.subheader("Explore the Data")
    st.dataframe(df)

# Visualization Section
elif sidebar_selection == "Visualization":
    st.subheader("Data Visualization")

    # Show the first few rows of the dataset
    st.write("Here is the first few rows of the data:")
    st.write(df.head())

    # Visualization options
    chart_type = st.selectbox("Select a chart type", ["Line Chart", "Bar Chart"])
    
    if chart_type == "Line Chart":
        st.line_chart(df['allowed_amount'])
    elif chart_type == "Bar Chart":
        st.bar_chart(df['allowed_amount'])

# Statistics Section
elif sidebar_selection == "Statistics":
    st.subheader("Data Statistics")

    st.write("Basic statistics for the dataset:")
    st.write(df.describe())

# Trends Section
elif sidebar_selection == "Trends":
    st.subheader("Trends Analysis")
    
    # Add your trends analysis logic here
    # Example: Identify trends in data
    st.write("Trends are analyzed based on historical data trends.")
