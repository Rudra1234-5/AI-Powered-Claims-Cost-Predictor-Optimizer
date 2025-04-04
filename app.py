import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from io import BytesIO

st.set_page_config(page_title="Healthcare Forecast App", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Gen_AI.csv")
    df['service_from_date'] = pd.to_datetime(df['service_from_date'], errors='coerce')
    return df

df = load_data()

date_column = 'service_from_date'
procedure_col = next((col for col in df.columns if 'procedure' in col.lower()), None)
drug_col = next((col for col in df.columns if 'ndc' in col.lower()), None)
diagnosis_col = next((col for col in df.columns if 'diagnosis' in col.lower()), None)
age_band_col = next((col for col in df.columns if 'age' in col.lower() and 'band' in col.lower()), None)

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

cost_column = st.sidebar.selectbox("Select cost column:", [col for col in df.columns if 'amount' in col.lower()])

if analysis_type == "Forecast":
    freq = 'Y'  # Force yearly forecast
    periods = st.sidebar.slider("Forecast years ahead", 1, 10, 3)

    df_grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[cost_column].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped[df_grouped['y'] > 0].dropna()

    if len(df_grouped) > 2:
        model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        model.fit(df_grouped)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        st.subheader("Forecasted Results (Yearly)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        forecast_df = forecast[['ds', 'yhat']]
        st.dataframe(forecast_df.tail(periods))

        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
    else:
        st.warning("Not enough yearly data to forecast.")

elif analysis_type == "Cost Distribution":
    group_column = st.sidebar.selectbox("Group by:", ['employee_gender', 'relationship', age_band_col])
    if group_column:
        grouped = df.groupby(group_column)[cost_column].sum().sort_values(ascending=False)
        st.subheader(f"Cost Distribution by {group_column}")
        st.bar_chart(grouped)
        st.dataframe(grouped)

elif analysis_type == "Per Employee Cost":
    df['year'] = df[date_column].dt.year
    grouped = df.groupby(['employee_id', 'year'])[cost_column].sum().reset_index()
    st.subheader("Per Employee Cost by Year")
    st.dataframe(grouped.head(100))

elif analysis_type == "Top 5 Diagnoses" and diagnosis_col:
    grouped = df.groupby(diagnosis_col)[cost_column].sum().nlargest(5)
    st.subheader("Top 5 Diagnoses by Cost")
    st.bar_chart(grouped)
    st.dataframe(grouped)

elif analysis_type == "Top 5 Drugs" and drug_col:
    grouped = df.groupby(drug_col)[cost_column].sum().nlargest(5)
    st.subheader("Top 5 Drugs by Cost")
    st.bar_chart(grouped)
    st.dataframe(grouped)

elif analysis_type == "Top 5 Costliest Claims This Month":
    latest_month = df[date_column].dt.to_period('M').max()
    subset = df[df[date_column].dt.to_period('M') == latest_month]
    st.subheader("Top 5 Costliest Claims This Month")
    st.dataframe(subset.nlargest(5, cost_column))
