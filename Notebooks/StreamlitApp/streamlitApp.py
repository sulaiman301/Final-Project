import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

plt.style.use("seaborn-dark-palette")

def oil_and_gas_eda_streamlit(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Display the first few rows
    st.write("### Dataset Preview")
    st.write(df.head())
    
    # Display the shape of the dataframe
    st.write(f"**Data shape:** {df.shape}")
    
    # Summary statistics
    st.write("### Summary Statistics")
    st.write(df.describe())
    
    # Data information
    st.write("### Data Information")
    st.write(df.info())
    
    # Visualizing Stock Trading Data
    st.write("### Trading Attributes Distribution")
    for y in ["Open", "High", "Volume"]:
        fig = px.histogram(df, x="Date", y=y, color="Symbol",
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           title=f"Total Trading {y} Distribution of Major US Oil Companies")
        fig.update_layout(template="plotly_dark", font=dict(family="PT Sans", size=18))
        st.plotly_chart(fig)
    
    # Visualizing All Attributes Together
    st.write("### All Stock Attributes Together")
    fig, ax = plt.subplots(figsize=(12, 12))
    df.plot(subplots=True, ax=ax, linewidth=1.5)
    plt.title("US Oil and Gas Stock Attributes")
    st.pyplot(fig)
    
    # Volume Trend Visualization
    st.write("### Volume Trend")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["Date"], df["Volume"])
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volume", fontsize=12)
    plt.title('Volume Trend', fontsize=20)
    plt.grid()
    st.pyplot(fig)
    
    # Moving Averages
    df["MA for 20 days"] = df["Open"].rolling(20).mean()
    df["MA for 50 days"] = df["Open"].rolling(50).mean()
    st.write("### Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    df.truncate(before="2010-01-01", after="2022-06-10")[["Close", "MA for 20 days", "MA for 50 days"]].plot(subplots=False, ax=ax, linewidth=2)
    plt.grid()
    st.pyplot(fig)
    
    # Distribution of Opening Prices
    st.write("### Distribution of Opening Prices")
    fig, ax = plt.subplots()
    sns.distplot(df["Open"], color="#FFD500", ax=ax)
    plt.title("Distribution of Open Prices of US Oil and Gas Stocks", fontweight="bold", fontsize=20)
    plt.xlabel("Open Price", fontsize=10)
    st.pyplot(fig)
    
    # Summary Statistics
    st.write("### Maximum and Minimum Opening Prices")
    st.write(f"Maximum open price of stock ever obtained: {df['Open'].max()}")
    st.write(f"Minimum open price of stock ever obtained: {df['Open'].min()}")
    
    # Missing Data Handling
    st.write("### Missing Data")
    missing_data = df.isnull().sum()
    st.write(missing_data)
    
    # Visualize missing data
    st.write("### Visualize Missing Data")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)
    
    # Drop rows with missing values
    df_cleaned = df.dropna()
    
    # Correlation Analysis
    st.write("### Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    plt.title("Correlation Matrix of Stock Attributes", fontsize=18)
    st.pyplot(fig)
    
    # Time Series Analysis
    st.write("### Time Series Decomposition")
    decomposed = seasonal_decompose(df['Close'], model='multiplicative', period=365)
    fig = decomposed.plot()
    plt.suptitle('Time Series Decomposition of Close Price', fontsize=18)
    st.pyplot(fig)
    
    # Comparison of Companies
    st.write("### Closing Price Trends by Company")
    fig = px.line(df, x="Date", y="Close", color="Symbol", title="Closing Price Trends of Major US Oil Companies")
    fig.update_layout(template="plotly_dark", font=dict(family="PT Sans", size=18))
    st.plotly_chart(fig)
    
    # Boxplot to Compare Volatilities
    st.write("### Volatility Comparison of Closing Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Symbol", y="Close", data=df, ax=ax)
    plt.title("Volatility Comparison of Closing Prices", fontsize=18)
    st.pyplot(fig)
    
    # Outlier Detection
    st.write("### Outlier Detection")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[['Open', 'High', 'Low', 'Close', 'Volume']], ax=ax)
    plt.title('Outlier Detection in Stock Attributes', fontsize=18)
    st.pyplot(fig)
    
    z_scores = np.abs(stats.zscore(df[['Open', 'High', 'Low', 'Close', 'Volume']]))
    outliers = np.where(z_scores > 3)
    st.write(f"Outliers Detected:\n{outliers}")
    
    # Interactive Selection
    st.write("### Interactive Comparison of Stock Prices")
    selected_symbols = st.multiselect(
        "Select Companies:",
        options=df['Symbol'].unique(),
        default=["XOM"]
    )
    
    if selected_symbols:
        filtered_df = df[df['Symbol'].isin(selected_symbols)]
        fig = px.line(filtered_df, x='Date', y='Close', color='Symbol')
        st.plotly_chart(fig)

# Run the Streamlit app
file_path = r"C:\Users\User\Documents\IRON HACK DA 2024\IH-Labs W8\Final-Project\Data\Raw data\oil and gas stock prices.csv"
st.title("Major US Oil and Gas Stock Price EDA")
oil_and_gas_eda_streamlit(file_path)
