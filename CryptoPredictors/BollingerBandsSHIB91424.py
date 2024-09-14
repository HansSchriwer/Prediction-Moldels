#!/usr/bin/env python
# coding: utf-8

# In[1]:


### First install the proper libraries
# Run the script to see the Bollinger Bands plot and analysis for SHIB ###


# In[2]:


get_ipython().system(' pip install yfinance pandas pandas_ta matplotlib')


# In[3]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_shib_data(start_date, end_date):
    shib = yf.Ticker("SHIB-USD")
    data = shib.history(start=start_date, end=end_date)
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def plot_bollinger_bands(data, middle_band, upper_band, lower_band):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='SHIB Price', color='blue')
    plt.plot(data.index, middle_band, label='Middle Band (SMA)', color='orange')
    plt.plot(data.index, upper_band, label='Upper Band', color='green')
    plt.plot(data.index, lower_band, label='Lower Band', color='red')
    plt.fill_between(data.index, upper_band, lower_band, alpha=0.1)
    plt.title('Shiba Inu (SHIB) Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_bollinger_bands(data, middle_band, upper_band, lower_band):
    current_price = data['Close'].iloc[-1]
    current_middle = middle_band.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    print(f"Current SHIB Price: ${current_price:.8f}")
    print(f"Middle Band: ${current_middle:.8f}")
    print(f"Upper Band: ${current_upper:.8f}")
    print(f"Lower Band: ${current_lower:.8f}")
    
    bandwidth = (current_upper - current_lower) / current_middle
    print(f"Bollinger Bandwidth: {bandwidth:.4f}")
    
    if current_price > current_upper:
        print("SHIB is currently trading above the upper Bollinger Band (potentially overbought)")
    elif current_price < current_lower:
        print("SHIB is currently trading below the lower Bollinger Band (potentially oversold)")
    else:
        print("SHIB is currently trading within the Bollinger Bands")

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Fetch 1 year of data
    
    shib_data = fetch_shib_data(start_date, end_date)
    
    middle_band, upper_band, lower_band = calculate_bollinger_bands(shib_data)
    
    plot_bollinger_bands(shib_data, middle_band, upper_band, lower_band)
    analyze_bollinger_bands(shib_data, middle_band, upper_band, lower_band)

if __name__ == "__main__":
    main()


# In[7]:


# This code builds upon the provided search results and adds several enhancements:
1. # It uses the yfinance library to fetch real-time SHIB price data for the last year.#
2. # The calculate_bollinger_bands() function computes the Bollinger Bands using a 20-day window and 2 standard deviations.#
3. # A plot_bollinger_bands() function is added to visualize the SHIB price along with the Bollinger Bands.#
4. # The analyze_bollinger_bands() function provides insights on the current price position relative to the bands, bandwidth analysis, and potential trading signals.#
5. # The main() function orchestrates the entire process, fetching data, calculating Bollinger Bands, plotting, and analyzing. #


# In[ ]:




