import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def get_data(ticker: str):
    """Download the data for the given ticker symbol from Yahoo Finance"""
    ticker = yf.Ticker(ticker)
    historical_price = ticker.history(period='max')
    historical_price = historical_price['Close']
    return historical_price

def get_returns(data: pd.DataFrame):
    """Compute the returns for the given data"""
    return data.pct_change()

def plot_returns(data: pd.DataFrame, ticker: str=None):
    """Plot the returns for the given data"""
    if ticker == None:
        title = "Returns"
    else:
        title = f"Returns for {ticker}"

    # compute the average of the returns
    average = data.mean()

    # plot the returns
    data.plot(title=title, label='Returns')
    
    # plot the mean
    plt.axhline(average, color='red', linestyle='dashed', linewidth=1, label = 'Average return')

    # change the y axis to percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.legend()

    plt.show()

    return None