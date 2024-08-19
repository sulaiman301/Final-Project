import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv(r"C:\Users\User\Documents\IRON HACK DA 2024\IH-Labs W8\Final-Project\Data\Raw data\oil and gas stock prices.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Volume'] = df['Volume'].astype(float)
    return df

df = load_data()

st.title("Major US Oil and Gas Stock Prices")

st.sidebar.header("Filters")

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Company selection
companies = df['Symbol'].unique()
selected_companies = st.sidebar.multiselect("Select companies", companies, default=companies)

df_filtered = df_filtered[df_filtered['Symbol'].isin(selected_companies)]

# Display filtered data
st.write(f"Displaying data from {start_date} to {end_date} for companies: {', '.join(selected_companies)}")

# Visualizations
st.header("Stock Price Visualization")

# Line plot of closing prices
fig = px.line(df_filtered, x='Date', y='Close', color='Symbol', title="Closing Price Trends")
st.plotly_chart(fig)

# Histogram of opening prices
fig = px.histogram(df_filtered, x='Open', color='Symbol', title="Distribution of Opening Prices")
st.plotly_chart(fig)

# Moving Averages
df_filtered['MA_20'] = df_filtered.groupby('Symbol')['Open'].transform(lambda x: x.rolling(20).mean())
df_filtered['MA_50'] = df_filtered.groupby('Symbol')['Open'].transform(lambda x: x.rolling(50).mean())

fig = px.line(df_filtered, x='Date', y=['Open', 'MA_20', 'MA_50'], color='Symbol', title="Moving Averages of Opening Prices")
st.plotly_chart(fig)

# Boxplot of closing prices
fig = px.box(df_filtered, x='Symbol', y='Close', title="Boxplot of Closing Prices")
st.plotly_chart(fig)

# Volume trend
fig = px.line(df_filtered, x='Date', y='Volume', color='Symbol', title="Trading Volume Over Time")
st.plotly_chart(fig)

# Correlation Heatmap
st.header("Correlation Analysis")
corr_matrix = df_filtered[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
st.pyplot(fig)

# Interactive Dashboard for Forecasting (Prophet Model)
st.header("Stock Price Forecasting")
from prophet import Prophet

df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet.dropna(inplace=True)

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

st.write("Forecasting for the next year:")
fig = m.plot(forecast)
st.pyplot(fig)

fig = m.plot_components(forecast)
st.pyplot(fig)

if __name__ == "__main__":
    st.write("Streamlit app is running...")
