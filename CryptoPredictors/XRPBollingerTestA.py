#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[7]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbol for XRP (Ripple)
ticker_symbol = "XRP-USD"

# Fetch historical data for XRP (Ripple)
xrp_data = yf.download(ticker_symbol, start="2023-05-01", end="2024-08-24")

# Calculate the 20-day moving average (SMA)
xrp_data['SMA'] = xrp_data['Close'].rolling(window=20).mean()

# Calculate the standard deviation over the same period
xrp_data['STD'] = xrp_data['Close'].rolling(window=20).std()

# Calculate the Upper and Lower Bollinger Bands
xrp_data['Upper Band'] = xrp_data['SMA'] + (2 * xrp_data['STD'])
xrp_data['Lower Band'] = xrp_data['SMA'] - (2 * xrp_data['STD'])

# Calculate the Bollinger Yield
xrp_data['Bollinger Yield'] = (xrp_data['Upper Band'] - xrp_data['Lower Band']) / xrp_data['SMA']

# Plot the Bollinger Bands along with the closing price
plt.figure(figsize=(14, 7))
plt.plot(xrp_data['Close'], label='XRP Close Price', color='blue')
plt.plot(xrp_data['Upper Band'], label='Upper Bollinger Band', color='red')
plt.plot(xrp_data['Lower Band'], label='Lower Bollinger Band', color='green')
plt.fill_between(xrp_data.index, xrp_data['Lower Band'], xrp_data['Upper Band'], color='lightgray')
plt.title('XRP Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper left')
plt.show()

# Show the most recent Bollinger Yield and Bands
bollinger_data = xrp_data[['Close', 'SMA', 'Upper Band', 'Lower Band', 'Bollinger Yield']].tail(60)

# Display the DataFrame in your environment
print(bollinger_data)


# In[5]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbol for XRP (Ripple)
ticker_symbol = "XRP-USD"

# Fetch historical data for XRP (Ripple)
xrp_data = yf.download(ticker_symbol, start="2023-05-01", end="2024-08-24")

# Calculate the 20-day moving average (SMA)
xrp_data['SMA'] = xrp_data['Close'].rolling(window=20).mean()

# Calculate the standard deviation over the same period
xrp_data['STD'] = xrp_data['Close'].rolling(window=20).std()

# Calculate the Upper and Lower Bollinger Bands
xrp_data['Upper Band'] = xrp_data['SMA'] + (2 * xrp_data['STD'])
xrp_data['Lower Band'] = xrp_data['SMA'] - (2 * xrp_data['STD'])

# Calculate the Bollinger Yield
xrp_data['Bollinger Yield'] = (xrp_data['Upper Band'] - xrp_data['Lower Band']) / xrp_data['SMA']

# Plot the Bollinger Bands along with the closing price
plt.figure(figsize=(14, 7))
plt.plot(xrp_data['Close'], label='XRP Close Price', color='blue')
plt.plot(xrp_data['Upper Band'], label='Upper Bollinger Band', color='red')
plt.plot(xrp_data['Lower Band'], label='Lower Bollinger Band', color='green')
plt.fill_between(xrp_data.index, xrp_data['Lower Band'], xrp_data['Upper Band'], color='lightgray')
plt.title('XRP Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper left')
plt.show()

# Show the most recent Bollinger Yield and Bands
bollinger_data = xrp_data[['Close', 'SMA', 'Upper Band', 'Lower Band', 'Bollinger Yield']].tail(60)
import ace_tools as tools; tools.display_dataframe_to_user(name="XRP Bollinger Data", dataframe=bollinger_data)

# Display the DataFrame in your environment
print(bollinger_data)


# In[ ]:




