
# Major US Oil and Gas Stock Price EDA
# Overview
This project conducts an Exploratory Data Analysis (EDA) on the stock prices of major US oil and gas companies. The analysis includes loading, cleaning, and visualizing the data, performing statistical analysis, identifying trends and outliers, and predicting future stock prices using the Prophet model. Additionally, an interactive dashboard is built using Dash to allow users to explore the data dynamically.

# Project Structure
data/: Contains the raw dataset (oil and gas stock prices.csv).
scripts/: Contains Python scripts including function.py for reusable functions.
notebooks/: Jupyter notebooks for step-by-step analysis.
README.md: This file, providing an overview of the project and how to run it.

# Installation

Requirements
To run the project, you will need the following Python libraries:

bash
Copy code
pandas
numpy
matplotlib
plotly
seaborn
statsmodels
scipy
dash
jupyter-dash
prophet

# Installation Instructions
Clone this repository to your local machine.
Create a virtual environment and activate it.
Install the required libraries by running:
bash
Copy code
pip install -r requirements.txt
Ensure you have Jupyter Notebook installed to run the notebooks.
# Dataset
The dataset used in this analysis contains stock price data for several major US oil and gas companies, including Exxon Mobil (XOM), Chevron (CVX), and ConocoPhillips (COP). It includes the following columns:

Date: The trading date.
Symbol: The stock symbol of the company.
Open: The opening price of the stock on that date.
High: The highest price of the stock on that date.
Low: The lowest price of the stock on that date.
Close: The closing price of the stock on that date.
Volume: The number of shares traded on that date.
Currency: The currency in which the stock is traded.

# Code Explanation
# 1. Importing Necessary Libraries
The project starts by importing essential Python libraries for data analysis, visualization, and modeling.

# 2. Data Loading and Exploration
Data Loading: The dataset is loaded into a pandas DataFrame.
Initial Exploration: Basic exploration is performed to understand the structure and contents of the dataset, including checking for missing values, data types, and summary statistics.

# 3. Data Visualization
Histograms: Histograms are used to visualize the distribution of trading attributes such as Open, High, and Volume.
Subplots: All attributes are plotted together using subplots to inspect trends over time.
Volume Trend: A bar plot is used to visualize the trend of trading volume over time.
Distribution Plots: Seaborn's distplot is used to visualize the distribution of opening prices. 
 
# 4. Moving Averages
Moving Averages Calculation: 20-day and 50-day moving averages are calculated for the Open price.
Plotting Moving Averages: The moving averages are plotted to observe trends over time.

# 5. Missing Data Handling
Check for Missing Data: Missing data is identified using a heatmap.
Handling Missing Data: Rows with missing values are dropped, and other methods like interpolation are suggested.

# 6. Correlation Analysis
Correlation Matrix: The correlation between different stock attributes is calculated and visualized using a heatmap to identify any significant relationships.

# 7. Time Series Analysis
Decomposition: The time series data is decomposed into trend, seasonality, and residuals using the statsmodels library.

# 8. Outlier Detection
Boxplot for Outliers: Boxplots are used to identify outliers in the stock attributes.
Z-scores for Outlier Detection: Z-scores are calculated to detect outliers, and decisions are made on whether to retain or remove them.

# 9. Interactive Dashboards with Dash
Dash App: An interactive dashboard is built using Dash, allowing users to select different companies and visualize their stock price trends over time.
Running the Dashboard: The Dash app is run within a Jupyter Notebook.

# 10. Future Predictions with Prophet
Prophet Model: The Prophet model is used to predict future stock prices. The model is fitted on the data, and future predictions are plotted.
# How to Run the Project
Run Jupyter Notebooks: Open the Jupyter notebooks in the notebooks/ directory to explore the EDA and analysis step by step.
Run Python Scripts: The scripts in the scripts/ directory contain reusable functions that can be executed to perform specific tasks.
Launch the Interactive Dashboard: Run the Dash app in the Jupyter notebook to interactively explore the stock data. The app will launch in your browser.
python
Copy code
app.run_server(mode='inline', port=8090)
Make Future Predictions: Use the Prophet model to forecast future stock prices. This can be run in the appropriate notebook or script.
Future Improvements
Additional Analysis: Consider adding more detailed analysis of the impact of external factors (e.g., oil prices, political events) on stock prices.
Advanced Modeling: Experiment with other time series forecasting models like ARIMA, LSTM, or XGBoost for improved predictions.
Interactive Elements: Enhance the Dash app with more interactive elements, such as filters for different date ranges or additional stock attributes.
# Conclusion
This project provides a comprehensive analysis of US oil and gas stock prices, combining EDA, visualization, and forecasting techniques. The use of both static and interactive visualizations allows for an in-depth exploration of the data, and the predictive modeling provides insights into potential future trends.

Contact
For any questions or suggestions, please contact [Your Name] at [Your Email].





