
import pandas as pd
from pmdarima import auto_arima
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



def test_stationarity(timeseries,title1="", namefile="", reg='', num=200):
    rolmean = timeseries.rolling(num).mean()
    rolstd = timeseries.rolling(num).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(title1)
    #plt.show(block=False)
    plt.savefig(namefile)
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test of %s:'%reg)
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100, namefile="", color1='tab:red'):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color=color1)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.savefig(namefile)
    plt.show()
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE


    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse })