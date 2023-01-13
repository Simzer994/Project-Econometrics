import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from arch import arch_model
from typing import Tuple
from scipy.optimize import minimize
from arch import arch_model


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


def weighted_hs_var(returns: pd.DataFrame,confidence_level: int, window: int,ticker: str=None)->pd.Series:
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
    else:
        title = f"Returns for {ticker} with Weighted HS VaR"
    
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

    # Plot the returns and the VaR on the same graph
    returns.plot(label='Returns')
    plt.plot(VaR, color='red', linestyle='dashed', linewidth=3, label = f'VaR {confidence_level}%')

    # change the bounds of the y axis
    plt.ylim(-0.3, 0.3)

    plt.title(label=title,fontweight='bold')

    plt.legend()

    plt.show()

    return VaR.VaR





def optimize_garch(returns: pd.DataFrame):
    """ Find the best parameters p and q using the log likelihood function """

    returns = returns.values

    # Define the objective function
    def objective_function(params: list):

        # Create the model
        model = arch_model(returns, p=params[0], q=params[1])

        # Fit the model
        model_fit = model.fit()

        # Return the negative log likelihood
        return -model_fit.loglikelihood

    # Define the bounds
    bounds = [(1, 5), (1, 5)]

    # Define the starting values
    starting_values = [1, 1]

    # Find the optimal parameters
    optimal_params = minimize(objective_function, starting_values, bounds=bounds)

    # Return the optimal parameters
    return optimal_params.x

