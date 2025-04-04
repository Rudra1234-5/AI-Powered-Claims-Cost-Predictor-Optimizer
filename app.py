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
    freq = st.sidebar.selectbox("Select forecast frequency", ['M', 'Q', 'Y'])
    periods = st.sidebar.slider("Forecast periods", 1, 36, 12)

    df_grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[cost_column].sum().reset_index()
    df_grouped.columns = ['ds', 'y']
    df_grouped = df_grouped[df_grouped['y'] > 0].dropna()

    if len(df_grouped) > 5:
        model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        model.fit(df_grouped)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        st.subheader("Forecasted Results")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        forecast_df = forecast[['ds', 'yhat']]
        st.dataframe(forecast_df.tail(periods))

        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
    else:
        st.warning("Not enough data to forecast.")

elif analysis_type == "Cost Distribution":
    dist_col = st.sidebar.selectbox("Select distribution parameter", [col for col in ['employee_gender', 'relationship', age_band_col] if col in df.columns and col is not None])
    grouped = df.groupby(dist_col)[cost_column].sum()

    st.subheader(f"Cost Distribution by {dist_col}")
    st.bar_chart(grouped)
    st.dataframe(grouped)

    csv = grouped.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"distribution_{dist_col}.csv", "text/csv")

elif analysis_type == "Per Employee Cost":
    grouped = df.groupby(['employee_id', df[date_column].dt.year])[cost_column].sum()
    st.subheader("Per Employee Cost by Year")
    st.dataframe(grouped)

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
    subset = df[df[date_column].dt.to_period('M') == latest_month][['claim_id', cost_column]]
    st.subheader("Top 5 Costliest Claims This Month")
    st.dataframe(subset.nlargest(5, cost_column))

elif analysis_type == "Inpatient vs Outpatient Costs" and 'type_of_service_description' in df.columns:
    grouped = df.groupby('type_of_service_description')[cost_column].sum()
    st.subheader("Inpatient vs Outpatient Costs")
    st.bar_chart(grouped)

elif analysis_type == "Chronic Disease %" and diagnosis_col:
    chronic = ['Diabetes', 'Hypertension']
    chronic_df = df[df[diagnosis_col].isin(chronic)]
    pct = (chronic_df['employee_id'].nunique() / df['employee_id'].nunique()) * 100
    st.subheader("Chronic Disease Prevalence")
    st.metric("% of Employees with Chronic Disease", f"{pct:.2f}%")

elif analysis_type == "Monthly Trends":
    grouped = df.groupby(df[date_column].dt.to_period('M'))[cost_column].sum()
    st.subheader("Monthly Cost Trends")
    st.line_chart(grouped)

elif analysis_type == "Year-over-Year Comparison":
    grouped = df.groupby(df[date_column].dt.year)[cost_column].sum()
    st.subheader("Year-over-Year Cost Comparison")
    st.bar_chart(grouped)

elif analysis_type == "Top 5 Hospitals by Spend" and 'hospital_name' in df.columns:
    grouped = df.groupby('hospital_name')[cost_column].sum().nlargest(5)
    st.subheader("Top 5 Hospitals by Spend")
    st.bar_chart(grouped)

elif analysis_type == "Average Cost by Employee Age Band" and age_band_col:
    grouped = df.groupby(age_band_col)[cost_column].mean()
    st.subheader("Average Cost by Employee Age Band")
    st.bar_chart(grouped)

elif analysis_type == "Total Claims by Provider Specialty" and 'provider_specialty_description' in df.columns:
    grouped = df['provider_specialty_description'].value_counts()
    st.subheader("Total Claims by Provider Specialty")
    st.bar_chart(grouped)

elif analysis_type == "Top 5 Service Types by Cost" and 'type_of_service_description' in df.columns:
    grouped = df.groupby('type_of_service_description')[cost_column].sum().nlargest(5)
    st.subheader("Top 5 Service Types by Cost")
    st.bar_chart(grouped)

elif analysis_type == "Most Common Diagnosis Categories" and diagnosis_col:
    grouped = df[diagnosis_col].value_counts().head(5)
    st.subheader("Most Common Diagnosis Categories")
    st.bar_chart(grouped)

elif analysis_type == "Cost Comparison by Gender" and 'employee_gender' in df.columns:
    grouped = df.groupby('employee_gender')[cost_column].mean()
    st.subheader("Cost Comparison by Gender")
    st.bar_chart(grouped)

elif analysis_type == "Top 5 Employers by Total Claims" and 'employer_name' in df.columns:
    grouped = df.groupby('employer_name')[cost_column].sum().nlargest(5)
    st.subheader("Top 5 Employers by Total Claims")
    st.bar_chart(grouped)

elif analysis_type == "Trend of Cost Over Time by Relationship" and 'relationship' in df.columns:
    grouped = df.groupby(['relationship', df[date_column].dt.to_period('M')])[cost_column].sum().unstack(0).fillna(0)
    st.subheader("Trend of Cost Over Time by Relationship")
    st.line_chart(grouped)

elif analysis_type == "In-Network vs Out-of-Network Spend" and 'network_indicator' in df.columns:
    grouped = df.groupby('network_indicator')[cost_column].sum()
    st.subheader("In-Network vs Out-of-Network Spend")
    st.bar_chart(grouped)

elif analysis_type == "Claim Spend by Place of Service" and 'place_of_service_description' in df.columns:
    grouped = df.groupby('place_of_service_description')[cost_column].sum().nlargest(5)
    st.subheader("Claim Spend by Place of Service")
    st.bar_chart(grouped)

else:
    st.warning("Unsupported option or missing column.")
