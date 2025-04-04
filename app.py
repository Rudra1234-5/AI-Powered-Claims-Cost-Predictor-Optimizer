import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def forecast_data(df, date_column, metric, periods, freq):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df_filtered = df.groupby(pd.Grouper(key=date_column, freq=freq))[metric].sum().reset_index()
    df_filtered.columns = ['ds', 'y']
    df_filtered = df_filtered[df_filtered['y'] > 0].dropna()
    
    if len(df_filtered) < 5:
        st.error(f"Not enough data for forecasting {metric}.")
        return None
    
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    model.fit(df_filtered)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def main():
    st.title("📊 GenAI Forecasting Chatbot")
    df = load_data()
    if df is None:
        st.warning("Please upload a CSV file to proceed.")
        return
    
    date_column = st.selectbox("Select Date Column", df.columns)
    cost_column = st.selectbox("Select Cost Column", [col for col in df.columns if 'amount' in col.lower()])
    analysis_type = st.selectbox("Select Analysis Type", [
        "Forecast", "Cost Distribution", "Top 5 Diagnoses", "Monthly Trends"
    ])
    
    if analysis_type == "Forecast":
        timeframe = st.selectbox("Select Frequency", ['M', 'Q', 'Y'])
        periods = st.number_input("Number of Future Periods", min_value=1, max_value=24, value=6)
        if st.button("Run Forecast"):
            forecast = forecast_data(df, date_column, cost_column, periods, timeframe)
            if forecast is not None:
                st.write("### Forecast Results")
                st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))
    
    elif analysis_type == "Cost Distribution":
        group_by = st.selectbox("Group By", [col for col in df.columns if col != cost_column])
        if st.button("Show Distribution"):
            st.write(df.groupby(group_by)[cost_column].sum().reset_index())
    
    elif analysis_type == "Top 5 Diagnoses" and 'diagnosis' in df.columns:
        if st.button("Show Diagnoses"):
            st.write(df.groupby('diagnosis')[cost_column].sum().nlargest(5))
    
    elif analysis_type == "Monthly Trends":
        if st.button("Show Trends"):
            df[date_column] = pd.to_datetime(df[date_column])
            trend = df.groupby(df[date_column].dt.to_period('M'))[cost_column].sum()
            st.line_chart(trend)
    
if __name__ == "__main__":
    main()
