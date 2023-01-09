import yfinance as yf
import pandas as pd

def get_data(ticker: str):
    """Download the data for the given ticker symbol from Yahoo Finance"""
    ticker = yf.Ticker(ticker)
    historical_price = ticker.history(period='max')
    historical_price = historical_price['Close']
    return historical_price

def get_returns(data: pd.DataFrame):
    """Calculate the returns for the given data"""
    return data.pct_change()
