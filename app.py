from utils import *
from openai import AzureOpenAI

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
                analysis_result = custom_analysis_with_ai(custom_query)
                st.write("AI Custom Analysis Result: ", analysis_result)
            else:
                st.error("Please enter a custom query for analysis.")
