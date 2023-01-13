import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from arch import arch_model
from typing import Tuple
from scipy.optimize import minimize
from arch import arch_model
from scipy.stats import norm
from arch.__future__ import reindexing
import numba


def get_data(ticker: str)->pd.DataFrame:
    """
    Download the data for the given ticker symbol from Yahoo Finance
    
    Parameters
    ----------
    ticker : str
        The ticker symbol of the stock
    
    Returns
    -------
    historical_price : pd.DataFrame
        The historical price of the stock
    """
    # Download the data using yahoo finance
    ticker = yf.Ticker(ticker)
    historical_price = ticker.history(period='max')

    # Keep only the closing price
    historical_price = historical_price['Close']

    # Select the period from 2007 to 2020
    historical_price = historical_price['2007-01-01':'2020-12-31']

    return historical_price


def get_returns(data: pd.DataFrame)->pd.Series:
    """
    Compute the returns for the given data
    
    Parameters
    ----------
    data : pd.DataFrame
        The data for which we want to compute the returns

    Returns
    -------
    returns : pd.Series
        The returns for the given data
    """
    # We drop the first line because it is NaN
    return data.pct_change().dropna()


def plot_returns(data: pd.DataFrame, ticker: str=None)->None:
    """
    Plot the returns for the given data
    
    Parameters  
    ----------
    data : pd.DataFrame
        The data for which we want to plot the returns
    ticker : str, optional
        The ticker symbol of the stock, by default None
    
    Returns
    -------
    None
    """
    if ticker == None:
        title = "Returns"
    else:
        title = f"Returns for {ticker}"

    # Compute the average of the returns
    average = data.mean()

    # Plot the returns
    data.plot(label='Returns')
    
    # plot the mean
    plt.axhline(average, color='red', linestyle='dashed', linewidth=3, label = 'Average return')

    # change the y axis to percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title(label=title,fontweight='bold')

    plt.legend()

    plt.show()

    return None


def weighted_hs_var(returns: pd.DataFrame,confidence_level: int, window: int,ticker: str=None, disp: bool=True)->pd.Series:
    """ 
    Estimation of the Value at Risk (VaR) using the Weighted Historical Simulation method with a rolling window

    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to estimate the VaR
    confidence_level : int
        The confidence level for which we want to estimate the VaR
    window : int
        The size of the rolling window
    ticker : str, optional
        The ticker symbol of the stock, by default None
    
    Returns
    -------
    VaR : pd.Series
        The estimated VaR for the given confidence level
    """
    if ticker == None:
        title = "Returns with Weighted HS VaR"
        titleVaR = "VaR with Weighted HS VaR"
    else:
        title = f"Returns for {ticker} with Weighted HS VaR"
        titleVaR = f"VaR for {ticker} with Weighted HS VaR"
    
    # Compute the rolling mean
    means = returns.rolling(window=window).mean()

    # Compute the rolling standard deviation
    weights = np.exp(-((returns - means)**2) / (2 * (means**2)))
    weights.rename('Weight', inplace=True)

    # Concatenate the returns and the weights
    merged = pd.concat([returns, weights], axis=1).dropna()

    VaR = pd.DataFrame(columns=['VaR'], index=merged.index)

    for i in range(merged.shape[0] - window + 1):

        # Select the returns and the weights for the current window
        current_returns = merged.iloc[i:i + window]['Close']
        current_weights = merged.iloc[i:i + window]['Weight']

        # Sort the returns and the weights in ascending order
        sorted_returns = current_returns.sort_values()
        sorted_weights = current_weights.sort_values(ascending=False)

        # Compute the index of the quantile corresponding to the confidence level
        quantile_index = int((confidence_level / 100) * current_returns.shape[0])

        # Select the return corresponding to the quantile index
        var = sorted_returns.iloc[quantile_index]

        # Store the VaR in the dataframe
        VaR.iloc[i + window - 1] = var

    if disp == True:

        # Plot the returns and the VaR on the same graph
        returns.plot(label='Returns')
        plt.plot(VaR, color='red', linestyle='dashed', linewidth=3, label = f'VaR {confidence_level}%')

        # change the bounds of the y axis
        plt.ylim(-0.3, 0.3)

        plt.title(label=title,fontweight='bold')

        plt.legend()

        plt.show()

        # Plot the VaR
        VaR.plot(label=f'VaR {confidence_level}%', linewidth=1, color='red')

        plt.title(label=titleVaR,fontweight='bold')

        plt.legend()

        plt.show()

        return VaR.VaR
    
    else:

        return VaR.VaR


def optimize_garch(returns: pd.DataFrame, bounds: list([int,int])):
    """
    Find the best parameters p and q using the log likelihood function
    
    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to find the best parameters
        
    Returns
    -------
    p,q : Tuple[int,int]
        The best parameters p and q
    """

    # Convert the returns to a numpy array
    returns = returns.values

    # Initialize the parameters p and q
    p,q=1,0

    model = arch_model(returns, p=1, q=0, dist='Normal', rescale=False)

    # Compute the log likelihood
    model_fit = model.fit(disp='off')

    likelihood = model_fit.loglikelihood   

    for i in range(1,bounds[0]):

        for j in range(0,bounds[1]):

            # Fit the GARCH model
            model = arch_model(returns, p=i, q=j, dist='Normal', rescale=False)

            # Compute the log likelihood
            model_fit = model.fit(disp='off')
            
            if model_fit.loglikelihood<likelihood:
                p,q=i,j
                likelihood=model_fit.loglikelihood
    
    return p,q


def garch_var(returns: pd.DataFrame, confidence_level: int, p: int, q: int, ticker: str=None, window: int=100, disp: bool=True)->pd.Series:
    """ 
    Estimation of the Value at Risk (VaR) using the GARCH method with a rolling window

    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to estimate the VaR
    confidence_level : int
        The confidence level for which we want to estimate the VaR
    bounds : list([int,int])
        The bounds for the parameters p and q
    ticker : str, optional
        The ticker symbol of the stock, by default None
    window : int
        The size of the rolling window
    
    Returns
    -------
    VaR : pd.Series
        The estimated VaR for the given confidence level
    """
    if ticker == None:
        title = "Returns with GARCH VaR"
        titleVaR = "VaR with GARCH"
    else:
        title = f"Returns for {ticker} with GARCH VaR"
        titleVaR = f"VaR for {ticker} with GARCH"

    # Initialize the dataframe for the VaR
    VaR = pd.DataFrame(columns=['VaR'], index=returns.index)

    for i in range(returns.shape[0] - window + 1):

        # Select the returns for the current window
        current_returns = returns.iloc[i:i + window]

        # Fit the GARCH model
        model = arch_model(current_returns, p=p, q=q, dist='Normal', rescale=False)
        model_fit = model.fit(disp='off')

        # Compute the VaR for the current window
        var = model_fit.forecast(horizon=1).variance.iloc[-1, 0] * norm.ppf((1-confidence_level/100))

        # Store the VaR in the dataframe
        VaR.iloc[i + window - 1] = - var

    # Plot the returns and the VaR on the same graph
    returns.plot(label='Returns')
    plt.plot(VaR, color='red', linestyle='dashed', linewidth=3, label = f'VaR {confidence_level}%')

    if disp == True:

        # change the bounds of the y axis
        plt.ylim(-0.3, 0.3)

        plt.title(label=title,fontweight='bold')

        plt.legend()

        plt.show()

        # Plot the VaR
        VaR.plot(label=f'VaR {confidence_level}%', linewidth=1, color='red')

        plt.title(label=titleVaR,fontweight='bold')

        plt.legend()

        plt.show()

        return VaR.VaR

    else:

        return VaR.VaR


def expected_shortfall(returns: pd.DataFrame, confidence_level: int, window: int=100, ticker: str=None, disp: bool=True)->pd.Series:
    """
    Estimation of the Expected Shortfall (ES) using a rolling window

    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to estimate the ES
    confidence_level : int
        The confidence level for which we want to estimate the ES
    window : int
        The size of the rolling window
    ticker : str, optional
        The ticker symbol of the stock, by default None

    Returns
    -------
    ES : pd.Series
        The estimated ES for the given confidence level
    """
    if ticker == None:
        title = "Returns with ES"
        titleES = "ES"
    else:
        title = f"Returns for {ticker} with ES"
        titleES = f"ES for {ticker}"
        
    # Compute the VaR
    VaR = weighted_hs_var(returns=returns, confidence_level=confidence_level, window=window, disp=False)

    # Initialize the dataframe for the ES
    ES = pd.DataFrame(columns=['ES'], index=returns.index)

    for i in range(returns.shape[0] - window + 1):

        # Select the returns for the current window
        current_returns = returns.iloc[i:i + window]

        # Select the Var for the current window
        var = VaR.iloc[i]

        # Compute the ES for the current window
        es = current_returns[current_returns > var].mean()

        # Store the ES in the dataframe
        ES.iloc[i + window - 1] = es

    if disp == True:

        # Plot the returns and the ES on the same graph
        returns.plot(label='Returns')
        plt.plot(ES, color='red', linestyle='dashed', linewidth=3, label = f'ES {confidence_level}%')

        # change the bounds of the y axis
        plt.ylim(-0.3, 0.3)

        plt.title(label=title,fontweight='bold')

        plt.legend()

        plt.show()

        # Plot the ES
        ES.plot(label = f'ES {confidence_level}%', linewidth=1, color='red')

        plt.title(label=titleES,fontweight='bold')

        plt.legend()

        plt.show()
    
        return ES.ES
    
    else:

        return ES.ES