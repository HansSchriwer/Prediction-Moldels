#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance pandas numpy matplotlib


# In[6]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbol for XRP
ticker = 'XRP-USD'

# Get historical market data from Yahoo Finance
xrp_data = yf.download(ticker, start='2017-01-01', end='2024-07-31')

# Calculate Bollinger Bands
def bollinger_bands(data, window=20):
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['20dSTD'] = data['Close'].rolling(window=window).std()
    data['Upper Band'] = data['MA20'] + (data['20dSTD'] * 2)
    data['Lower Band'] = data['MA20'] - (data['20dSTD'] * 2)
    return data

xrp_data = bollinger_bands(xrp_data)

# Calculate RSI
def rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

xrp_data = rsi(xrp_data)

# Calculate MACD
def macd(data, slow=26, fast=12, signal=9):
    data['EMA12'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    return data

xrp_data = macd(xrp_data)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# Bollinger Bands plot
ax1.plot(xrp_data.index, xrp_data['Close'], label='Close Price', color='blue')
ax1.plot(xrp_data.index, xrp_data['MA20'], label='20 Day MA', color='red')
ax1.plot(xrp_data.index, xrp_data['Upper Band'], label='Upper Band', color='green')
ax1.plot(xrp_data.index, xrp_data['Lower Band'], label='Lower Band', color='green')
ax1.set_title('Bollinger Bands')
ax1.legend()

# RSI plot
ax2.plot(xrp_data.index, xrp_data['RSI'], label='RSI', color='purple')
ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
ax2.set_title('Relative Strength Index (RSI)')
ax2.legend()

# MACD plot
ax3.plot(xrp_data.index, xrp_data['MACD'], label='MACD', color='black')
ax3.plot(xrp_data.index, xrp_data['Signal Line'], label='Signal Line', color='red')
ax3.bar(xrp_data.index, xrp_data['MACD'] - xrp_data['Signal Line'], label='MACD Histogram', color='grey')
ax3.set_title('Moving Average Convergence Divergence (MACD)')
ax3.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




