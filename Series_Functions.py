import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import norm
from arch import arch_model
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

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


def find_best_garch_param(returns):
    """
    Function to find best GARCH(p,q) parameters using grid search with cross validation.
    Parameters:
        - returns: the return series
    Returns:
        - best_params: the best parameters (p,q) for the GARCH model
    """
    p_values = range(0,5)
    q_values = range(0,5)
    best_aic = np.inf
    best_order = None
    best_mdl = None
    for i in p_values:
        for j in q_values:
            try:
                tmp_mdl = arch_model(returns, vol='GARCH', p=i, o=0, q=j)
                res = tmp_mdl.fit(update_freq=5)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (i, j)
                    best_mdl = tmp_mdl
            except: continue
    return best_order


def estimate_var_garch(returns, confidence_level=0.05, horizon=1):
    """
    Estimates Value-at-Risk (VaR) using GARCH method.
    Parameters:
        - returns: the return series
        - confidence_level: the level of confidence, default is 0.05
        - horizon : the horizon of the forecast, default is 1
    Returns:
        - VaR: the estimated VaR
    """
    # Build GARCH model
    model = arch_model(returns, mean='Constant', vol='GARCH', p=1, o=0, q=1)

    # Fit the model to the data
    fit = model.fit()

    # Estimate VaR at given confidence level
    VaR = fit.var_forecast(horizon=horizon).mean['h.1'] * np.sqrt(horizon)
    return VaR

