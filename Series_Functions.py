import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from arch import arch_model
from typing import Tuple
from scipy.optimize import minimize
from arch import arch_model


def get_data(ticker: str):
    """Download the data for the given ticker symbol from Yahoo Finance"""

    # download the data using yahoo finance
    ticker = yf.Ticker(ticker)
    historical_price = ticker.history(period='max')

    # keep only the closing price
    historical_price = historical_price['Close']

    return historical_price


def get_returns(data: pd.DataFrame):
    """Compute the returns for the given data"""
    return data.pct_change().dropna()


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


def weighted_hs_var(returns: pd.DataFrame, confidence_level: int):
    """ Estime la VaR en utilisant la méthode "Weighted HS" """

    # Calculer la moyenne des rendements
    mean = returns.mean()

    # Créer un tableau de poids en fonction de la distance de chaque rendement par rapport à la moyenne
    # (ici, nous utilisons la fonction gaussienne comme kernel)
    weights = np.exp(-((returns - mean)**2) / (2 * (mean**2)))

    # Trier les données de rendement et leurs poids correspondants en ordre croissant et décroissant
    sorted_returns = returns.sort_values()
    sorted_weights = weights.sort_values(ascending=False)

    # Calculer l'indice du quantile correspondant au niveau de confiance choisi
    quantile_index = int((confidence_level / 100) * returns.shape[0])

    # Sélectionner la valeur de rendement correspondante dans le tableau de rendements trié
    var = sorted_returns.iloc[quantile_index]

    # Retourner la VaR
    print(f"VaR au niveau de confiance {confidence_level}% : {var:.4f}")

    return var

def optimize_garch(returns: pd.DataFrame):
    """ Find the best parameters p and q using the log likelihood function """

    returns = returns.values

    # Define the objective function
    def objective_function(params):

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

