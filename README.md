# Forecasting-Net-Prophet
  First, imported all necessary libraries & dependencies to be used in Jupyter Lab for this assignment:

  !pip install pystan
  !pip install fbprophet
  !pip install hvplot
  !pip install holoviews

import pandas as pd
import holoviews as hv
from fbprophet import Prophet
import hvplot.pandas
import datetime as dt
%matplotlib inline

I attempted at the 5 steps, last one was optional:

Step 1: Find Unusual Patterns in Hourly Google Search Traffic

Step 2: Mine the Search Traffic Data for Seasonality

Step 3: Relate the Search Traffic to Stock Price Patterns

Step 4: Create a Time Series Model with Prophet

Step 5 (Optional): Forecast Revenue by Using Time Series Models

In step 1, 
The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.
To do so, complete the following steps:
1. Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?
2. Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?
I uploaded  the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame

Read the ‘google_hourly_search_trends.csv to review the head / tail of the file.

Here, I had an error come up in Jupyter Lab and didn’t have an error in Colab. It went through with no issue.

![Alt text](//Users/jerilacson/Desktop/img.jpg?raw=true "Optional Title")



Overall, it was to find out about when google search traffic increases and decreases..

In step 2, it was to ‘Mine the Search traffic For Seasonality’ 
Finding out which day of the week can we predicts seasonal patterns of interest in the company.

Grouped the hourly search data to plot the average traffic by the day of the week.

Used hvPlot to visualise this traffic as a heat map.  Looking at any day of the week, are there particular hours of the day with the highest traffic?

During the holiday season, does the traffic increase?
There is an increase upward towards week 50 and then the last 2 weeks there’s a decrease.

In step 3, ‘Relate the Search Traffic to Stock Price Patterns’.

Read and plot the stock price data.csv

1. Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
    * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
    * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
2. Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

In step 4, ‘Create a Time Series Model with Prophet’
Produce a time series model that analyzes and forecasts patterns in the hourly search data. 

1. Set up the Google search data for a Prophet forecasting model.
2. After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?
3. Plot the individual time series components of the model to answer the following questions:
    * What time of day exhibits the greatest popularity?
    * Which day of the week gets the most search traffic?
    * What's the lowest point for search traffic in the calendar year?
In Step 5 (Optional): 
Forecast Revenue by Using Time Series Models
Step 1: Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.
In [ ]:
# Upload the "mercado_daily_revenue.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the DatetimeIndex
# Sales are quoted in millions of US dollars
from google.colab import files
uploaded = files.upload()

df_mercado_sales = pd.read_csv(
    'mercado_daily_recenue.csv',
    index_col='date',
    parse_dates=True,
    infer_datetime_format=True
)

# Read df_mercado_sales

df_mercado_sales = pd.read_csv(
    Path('Resources/mercado_daily_revenue.csv',
         index_col='date', parse_dates=True,
         infer_datetime_format=True)
)
# Review the DataFrame
display(df_mercado_sales)

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the daily sales figures 
df_mercado_sales.hvplot(title = 'Daily Sales Figures - MercadoLibre',rot=90)
In [ ]:

# Apply a Facebook Prophet model to the data.

# Set up the dataframe in the neccessary format:
# Reset the index so that date becomes a column in the DataFrame
mercado_sales_prophet_df = df_mercado_sales.reset_index()

# Adjust the columns names to the Prophet syntax
mercado_sales_prophet_df.columns = ['ds','y']

# Visualize the DataFrame
mercado_sales_prophet_df.head()

 Create the model
mercado_sales_prophet_model = Prophet()

# Fit the model
mercado_sales_prophet_model.fit(mercado_sales_prophet_df)

Step 2: Interpret the model output to identify any seasonal patterns in the company's revenue. Question:
For example, what are the peak revenue days? (Mondays? Fridays? Something else?)


Answer: Wednesday's










