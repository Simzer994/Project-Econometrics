import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
from arch import arch_model
from arch.__future__ import reindexing
from arch.univariate import arch_model, ConstantMean, GARCH, Normal,StudentsT
from typing import Tuple
from scipy.optimize import minimize
from scipy.stats import norm,t,skew,kurtosis,jarque_bera
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statistics import NormalDist
import os


def get_data(ticker: str,start:str='2007-01-01',end:str='2022-01-01')->pd.DataFrame:
    """
    Download the data for the given ticker symbol from Yahoo Finance
    
    Parameters
    ----------
    ticker : str
        The ticker symbol of the stock
    start : str, optional
        The start date of the period, by default '2007-01-01'
    end : str, optional
        The end date of the period, by default '2022-01-01'
    
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

    # Select the period from start to end
    historical_price = historical_price[start:end]

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
    
    # Plot the mean
    plt.axhline(average, color='red', linestyle='dashed', linewidth=3, label = 'Average return')

    # Change the y axis to percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    # Axis labels
    plt.xlabel('Dates',fontweight='bold')
    plt.ylabel('Returns',fontweight='bold')

    plt.title(label=title,fontweight='bold')

    plt.legend()

    plt.show()

    return None

def weighted_hs_var(returns:pd.DataFrame,confidence_level:int=95,window:int=250,ticker:str=None,disp:bool=True,l:float=0.96)->pd.DataFrame:
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
    disp : bool, optional
        If True, the returns and the VaR are plotted on the same graph, by default True
    l : float, optional
        The lambda parameter for the Age-weighted, by default 0.96
    
    Returns
    -------
    VaR : pd.DataFrame
        The HS-VaR and Age-Weighted HS-VaR for the given returns
    """
    if ticker == None:
        title = "Returns with Weighted HS VaR"
        titleVaR = "VaR with Weighted HS VaR"
    else:
        title = f"Returns for {ticker} with Weighted HS VaR"
        titleVaR = f"VaR for {ticker} with Weighted HS VaR"

    VaR = pd.DataFrame(columns=['HS-VaR',"AWHS-VaR"], index=returns.index)

    for i in range(returns.shape[0] - window + 1):

        # Select the returns and the weights for the current window
        current = pd.DataFrame(returns.iloc[i:i + window].values,columns=["Returns"]).reset_index(drop=True)
        current["Weights"] = np.sort([((1-l)*l**(i-1))/(1-l**window) for i in range(1,window+1)])

        # Sort the returns and the weights in non-ascending order and compute the cumulative sum of the weights

        sorted_current = current.sort_values(by="Returns",ascending=False)
        sorted_current["Cumsum"] = sorted_current["Weights"].cumsum()
        sorted_current = sorted_current.reset_index(drop=True)

        # Compute the index of the quantile corresponding to the confidence level
        quantile_index = int((confidence_level / 100) * sorted_current.shape[0])
        agquantile_index = ((sorted_current['Cumsum'] > 0.95)).idxmax()

        # Select the return corresponding to the quantile index
        hsvar = sorted_current["Returns"].iloc[quantile_index]
        aghsvar = sorted_current["Returns"].iloc[agquantile_index]

        # Store the VaR in the dataframe
        VaR['HS-VaR'].iloc[i + window - 1] = hsvar 
        VaR['AWHS-VaR'].iloc[i + window - 1] = aghsvar

    if disp == True:

        # Plot the returns and the VaR on the same graph
        returns.plot(label='Returns')
        plt.plot(VaR['HS-VaR'], color='red', linewidth=1.5, label = f'HS-VaR {confidence_level}%')
        plt.plot(VaR['AWHS-VaR'], color='black', linewidth=1.5, label = f'Age-Weighted HS-VaR {confidence_level}%')

        # change the bounds of the y axis
        plt.ylim(-0.3, 0.3)

        plt.title(label=title,fontweight='bold')

        plt.xlabel('Dates',fontweight='bold')
        plt.ylabel('Returns',fontweight='bold')

        plt.legend()

        plt.show()

        # Plot the VaR
        plt.plot(VaR['HS-VaR'], color='red', linewidth=1.5, label = f'HS-VaR {confidence_level}%')
        plt.plot(VaR['AWHS-VaR'], color='black', linewidth=1.5, label = f'Age-Weighted HS-VaR {confidence_level}%')

        plt.title(label=titleVaR,fontweight='bold')

        plt.xlabel('Dates',fontweight='bold')
        plt.ylabel('VaR',fontweight='bold')

        plt.legend()

        plt.show()

        return VaR
    
    else:

        return VaR


### USELESS FUNCTIONS ###


def optimize_garch(returns: pd.DataFrame, bounds: list([int,int,int])):
    """
    Find the best parameters p, d and q using the log likelihood function
    
    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to find the best parameters
        
    Returns
    -------
    p,d,q : Tuple[int,int,int]
        The best parameters p, d and q
    """

    # Convert the returns to a numpy array
    returns = returns.values

    # Initialize the parameters p and q
    p,d,q=1,0,0

    model = arch_model(returns, p=1,q=0, dist='Normal', rescale=False)

    # Compute the log likelihood
    model_fit = model.fit(disp='off')

    likelihood = model_fit.loglikelihood   

    for a in range(1,bounds[0]):

        for b in range(0,bounds[1]):

            for c in range(0,bounds[2]):

                # Fit the GARCH model
                model = arch_model(returns,p=a,o=b,q=c,dist='Normal',rescale=False)

                # Compute the log likelihood
                model_fit = model.fit(disp='off')
                
                if model_fit.loglikelihood<likelihood:
                    p,d,q
                    likelihood=model_fit.loglikelihood
        
    return p,d,q


def garch_var(returns: pd.DataFrame,p:int,q:int,d:int,confidence_level:int=0.95,ticker:str=None,window:int=250,disp:bool=True)->pd.Series:
    """ 
    Estimation of the Value at Risk (VaR) using the GARCH method with a rolling window

    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to estimate the VaR
    p : int
        The order of the AR part of the GARCH model
    q : int
        The order of the MA part of the GARCH model
    d : int
        The order of the differencing part of the GARCH model
    confidence_level : int
        The confidence level for the VaR
    ticker : str
        The ticker of the asset for which we want to estimate the VaR
    window : int
        The size of the rolling window
    disp : bool 
        If True, the returns and the VaR are plotted on the same graph
    
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


## END OF USELESS FUNCTIONS ##


def garch_var2(returns: pd.DataFrame,confidence_level:int=95,ticker:str=None,window:int=250,disp:bool=True)->pd.Series:
    """ 
    Estimation of the Value at Risk (VaR) using the GARCH method with a rolling window

    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to estimate the VaR
    confidence_level : int
        The confidence level for the VaR
    ticker : str
        The ticker of the asset for which we want to estimate the VaR
    window : int
        The size of the rolling window
    disp : bool 
        If True, the returns and the VaR are plotted on the same graph
    
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
    sigma2 = pd.DataFrame(columns=['sigma_2'], index=returns.index)

    for i in range(returns.shape[0] - window + 1):

        # Select the returns for the current window
        current_returns = returns.iloc[i:i + window]

        # Create the GARCH model 
        model = arch_model(current_returns)
        model = ConstantMean(current_returns)
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = Normal()
        model = model.fit(disp='off')
        aic = model.aic

        # Choose between a Normal or a Student-t distribution
        model2 = arch_model(current_returns)
        model2 = ConstantMean(current_returns)
        model2.volatility = GARCH(p=1, o=0, q=1)
        model2.distribution = StudentsT()
        model2 = model2.fit(disp='off')
        aic2 = model2.aic
        student = False

        if aic2 < aic:
            model = model2
            student = True

        if i == 0:
                unc_variance = (model.params[1] / (1 - model.params[2] - model.params[3]))
                sigma2.iloc[i+window-1] = model.params[1] + model.params[2]*(current_returns.iloc[window-1] - model.params[0])**2 + model.params[3]*unc_variance
        else:
                
            sigma2.iloc[i+window-1] = model.params[1] + model.params[2]*(current_returns.iloc[window-1] - model.params[0])**2 + model.params[3]*sigma2.iloc[i+window-2] 

        if student:
            nu = round(model.params['nu'])
            VaR.iloc[i+window-1] = model.params[0] + math.sqrt(sigma2.iloc[i+window-1])*t.ppf(1-confidence_level/100, nu)
        
        else:
            VaR.iloc[i+window-1] = model.params[0] + math.sqrt(sigma2.iloc[i+window-1])*NormalDist(mu=0, sigma=1).inv_cdf(1-confidence_level/100)

    if disp == True:

        # Plot the returns and the VaR on the same graph
        returns.plot(label='Returns')
        plt.plot(VaR, color='red', linestyle='dashed', linewidth=3, label = f'VaR {confidence_level}%')

        # Change the bounds of the y axis
        plt.ylim(-0.3, 0.3)

        plt.title(label=title,fontweight='bold')

        plt.xlabel('Dates',fontweight='bold')
        plt.ylabel('Returns',fontweight='bold')

        plt.legend()

        plt.show()

        # Plot the VaR
        VaR.plot(label=f'VaR {confidence_level}%', linewidth=1, color='red')

        plt.title(label=titleVaR,fontweight='bold')

        plt.xlabel('Dates',fontweight='bold')
        plt.ylabel('VaR',fontweight='bold')

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
        var = VaR.iloc[i]["AWHS-VaR"]

        # Compute the ES for the current window
        es = current_returns[current_returns <= var].mean()

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

def Dickey_Fuller(returns: pd.DataFrame, ticker: str=None,disp:bool=True)->float:
    """ 
    Dickey-Fuller test for unit root

    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to test the unit root
    ticker : str, optional
        The ticker symbol of the stock, by default None
    disp : bool, optional
        If True, the results of the test are printed, by default True

    Returns
    -------
    float
        The p-value of the test
    """

    # transform the returns into a numpy array
    returns = returns.values

    # make the Dickey-Fuller test
    test = adfuller(returns)

    # Printing the statistical result of the adfuller test

    if disp:
        if ticker == None:
            print('Augmented Dickey_fuller Statistic: %f' % test[0])
            print('p-value: %f' % test[1])
        else:
            print(f'Augmented Dickey_fuller Statistic for {ticker}: %f' % test[0])
            print('p-value: %f' % test[1])
        
        # printing the critical values at different alpha levels.
        print('Critical values at different levels:')
        for k, v in test[4].items():
            print('\t%s: %.3f' % (k, v))

    return test[1]


def plot_comparison(VaR_NonParam: pd.DataFrame, VaR_Param: pd.DataFrame, ES: pd.DataFrame=None, ticker: str=None, confidence_level: int=95):
    """
    Plot the VaR and ES for the non-parametric and parametric methods

    Parameters
    ----------
    VaR_NonParam : pd.DataFrame
        The VaR for the non-parametric method
    VaR_Param : pd.DataFrame
        The VaR for the parametric method
    ES : pd.DataFrame
        The ES for the parametric method
    ticker : str, optional
        The ticker symbol of the stock, by default None
    confidence_level : int, optional
        The confidence level for which we want to plot the VaR and ES, by default 95
    """
    if ticker == None:
        title = f"VaR and ES for the non-parametric and parametric methods with a confidence level of {confidence_level}%"
    else:
        title = f"VaR and ES for the non-parametric and parametric methods for {ticker} with a confidence level of {confidence_level}%"

    # Plot the HS-VaR for the non-parametric method
    VaR_NonParam["HS-VaR"].plot(label=f'Historical Simulation VaR {confidence_level}%', linewidth=1, color='red')

    VaR_NonParam["AWHS-VaR"].plot(label=f'Age Weighted Historical Simulation VaR {confidence_level}%', linewidth=1, color='black')

    # Plot the VaR for the parametric method
    VaR_Param.plot(label=f'GARCH VaR {confidence_level}%', linewidth=1, color='blue')

    if ES is not None:
        # Plot the ES for the parametric method
        ES.plot(label=f'ES {confidence_level}%', linewidth=1, color='green')

    plt.title(label=title,fontweight='bold')

    plt.xlabel('Dates',fontweight='bold')
    plt.ylabel('VaR and ES',fontweight='bold')

    plt.legend()

    plt.show()


def download_dataFrame(datas: dict, ticker: str):
    """
    Download the dataFrame as a csv file

    Parameters
    ----------
    data : dict
        The dict of dataFrame to download
    ticker : str
        The ticker symbol of the stock
    """
    if not os.path.exists('./datas/'+ticker):
        os.makedirs('./datas/'+ticker)
    for key, value in datas.items():
        value.to_csv('./datas/'+ticker+'/'+key+'.csv')
        
def dev_from_normality(returns: pd.DataFrame, ticker: str=None, disp: bool=True):
    """
    Test the deviation from normality with the Jarque Bera test, the Kurtosis and the Skewness
    
    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to test the unit root
    ticker : str, optional
        The ticker symbol of the stock, by default None
    disp : bool, optional
        If True, the results of the test are printed, by default True
    """
    
    print(f"Skewness of the returns: {returns.skew()}")
    print(f"Kurtosis of the returns: {returns.kurtosis()}")
    print(f"Jarque Bera test: {jarque_bera(returns)[0]} with a p-value of {jarque_bera(returns)[1]}")
    return None

def autocorr(returns: pd.DataFrame, ticker: str=None, lags=20): 
    """
    Plot the ACF of returns and squarred returns with the p-value of the Durbin-Watson test
    
    Parameters
    ----------
    returns : pd.DataFrame
        The returns for which we want to test the unit root
    ticker : str, optional
        The ticker symbol of the stock, by default None
    lags : int, optional   
    """
    # transform the returns into a numpy array
    returns = returns.values
    squarred_returns = returns**2
    
    if ticker == None:
        title = f"Autocorrelation function of the returns"
        title2 = f"Autocorrelation function of the squarred returns"
    else:
        title = f"Autocorrelation function of the returns for {ticker}"
        title2 = f"Autocorrelation function of the squarred returns for {ticker}"

    # Plot the ACF of returns
    sm.graphics.tsa.plot_acf(returns, lags=lags, title=title)

    # Plot the ACF of squarred returns
    sm.graphics.tsa.plot_acf(squarred_returns, lags=lags, title=title2)
    
    # Print the value of the Durbin-Watson test
    print(f"Durbin-Watson test for the returns: {sm.stats.stattools.durbin_watson(returns)}")
    print(f"Durbin-Watson test for the squarred returns: {sm.stats.stattools.durbin_watson(squarred_returns)}")
