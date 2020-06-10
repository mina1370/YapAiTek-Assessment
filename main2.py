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
from util import test_stationarity, plot_df, forecast_accuracy


df = pd.read_csv('psi_df_2016_2019.csv', parse_dates=['timestamp'])
df.index = df['timestamp']
daily_summary = pd.DataFrame()
monthly_summary = pd.DataFrame()
region=['west','east','south','north','central','national']

for reg in region:
    daily_summary[reg] = df[reg].resample('D').mean()
    monthly_summary[reg] = df[reg].resample('M').mean()



monthly_summary['timestamp']=monthly_summary.index
monthly_summary=monthly_summary.fillna(method='ffill')
monthly_summary=monthly_summary.fillna(method='bfill')
monthly_summary=monthly_summary.dropna()
daily_summary['timestamp']=daily_summary.index
daily_summary=daily_summary.fillna(method='ffill')
daily_summary=daily_summary.fillna(method='bfill')
daily_summary=daily_summary.dropna()

a = 1

if a == 1:
    for reg in region:
        plot_df(daily_summary, x=daily_summary.timestamp, y=daily_summary[reg],
                title='daily Pollutant Standards Index (PSI) in %s Singapore from 2016 to 2020.'% reg,
                namefile='Fig/%s_daily Pollutant'% reg,color1='tab:red')
        plot_df(df, x=df.timestamp, y=df[reg],
                title='hourly Pollutant Standards Index (PSI) in %s Singapore from 2016 to 2020.' % reg,
                namefile='Fig/%s_hourly Pollutant' % reg, color1='tab:blue')
        plot_df(monthly_summary, x=monthly_summary.timestamp, y=monthly_summary[reg],
                title='monthly Pollutant Standards Index (PSI) in %s Singapore from 2016 to 2020.' % reg,
                namefile='Fig/%s_monthly Pollutant' % reg, color1='tab:green')


daily_summary['year'] = [d.year for d in daily_summary.timestamp]
daily_summary['day'] = [d.day for d in daily_summary.timestamp]
daily_summary['month'] = [d.strftime('%b') for d in daily_summary.timestamp]
daily_summary['day_of_week']=daily_summary.timestamp.dt.dayofweek

b=1
if b == 1:
    for reg in region:
#
        fig, axes = plt.subplots(1, 3, figsize=(20,7), dpi= 80)
        sns.boxplot(x='year', y=reg, data=daily_summary, ax=axes[0])
        sns.boxplot(x='day_of_week', y=reg, data=daily_summary, ax=axes[1])
        sns.boxplot(x='month', y=reg, data=daily_summary.loc[~daily_summary.year.isin([2016, 2020]), :])
#
#         # Set Title
        axes[0].set_title('Year-wise Box Plot \n %s' %reg, fontsize=18)
        axes[1].set_title('day_of_week Box Plot \n %s' %reg, fontsize=18)
        axes[2].set_title('Month-wise Box Plot \n %s' %reg, fontsize=18)
        plt.savefig('Fig/Box Plot %s' %reg)
        plt.show()
        fig= plt.subplots(1, 1, figsize=(20,7), dpi= 80)
        sns.boxplot(x='day', y=reg, data=daily_summary )
        plt.title('day_of Month Box Plot \n %s' %reg, fontsize=18)
        plt.savefig('Fig/Box Plot day %s' %reg)
        plt.show()
# #
c = 1
if c==1:
    for reg in region:
        test_stationarity(daily_summary[reg].dropna(), title1='Rolling Mean & Standard Deviation \n %s' % reg,
                          namefile='Fig/ stationarity %s' % reg , reg=reg , num=200)
        test_stationarity(monthly_summary[reg].dropna(), title1='Rolling Mean & Standard Deviation monthly \n %s' % reg,
                          namefile='Fig/ stationarity_monthly %s' % reg , reg=reg , num=10)
        test_stationarity(monthly_summary[reg].diff().dropna(), title1='Rolling Mean & Standard Deviation monthly diff \n %s' % reg,
                          namefile='Fig/ stationarity_monthly_diff %s' % reg , reg=reg , num=10)

d = 1
if d==1:
    for reg in region:
        b = seasonal_decompose(daily_summary[reg], model="add", freq=30 * 12)
        b.plot()
        plt.savefig('Fig/seasonal decompose 3012 %s' % reg)
        plt.show()
        b = seasonal_decompose(daily_summary[reg], model="add", freq=30 )
        b.plot()
        plt.savefig('Fig/seasonal decompose 30 %s' % reg)
        plt.show()
        b = seasonal_decompose(daily_summary[reg], model="add", freq=7)
        b.plot()
        plt.savefig('Fig/seasonal decompose 7 %s' % reg)
        plt.show()

e = 1
if e==1:
    for reg in region:
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_pacf(daily_summary[reg].diff().dropna(), lags=40, ax=ax1 )
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_acf(daily_summary[reg].diff().dropna(), lags=40, ax=ax2 )
        plt.savefig('Fig/ACF and PACF %s' % reg)
        plt.show()
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_pacf(monthly_summary[reg].diff().dropna(), lags=40, ax=ax1 )
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_acf(monthly_summary[reg].diff().dropna(), lags=40, ax=ax2 )
        plt.savefig('Fig/ACF and PACF monthly %s' % reg)
        plt.show()



# # prediction


for reg in region:
    mod = SARIMAX(daily_summary[reg], trend='n', order=(4,0,2), seasonal_order=(0,0,0,7))
    results = mod.fit()
    print(results.summary())
    daily_summary['forecast'] = results.predict(start = 1200, end= 1369, dynamic= True)
    daily_summary['onestepforcast'] = results.predict(start = 1200, end= 1369, dynamic= False)
    daily_summary[[reg, 'forecast','onestepforcast']].plot(figsize=(12, 8))
    plt.title('ARIMA method')
    plt.savefig('Fig/forcast_nonseason %s' % reg)
    plt.show()
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals \n ARIMA method", ax=ax[0])
    residuals.plot(kind='kde', title='Density \n ARIMA method', ax=ax[1])
    plt.savefig('Fig/residual_nonseason %s' % reg)
    plt.show()
    ac1 = forecast_accuracy(daily_summary['forecast'][1200:1369], daily_summary[reg][1200:1369])
    ac2 = forecast_accuracy(daily_summary['onestepforcast'][1200:1369], daily_summary[reg][1200:1369])
    print('%s nonseason forecast' % reg)
    print(ac1)
    print('%s nonseason onestepforcast' % reg)
    print(ac2)

    mod = SARIMAX(daily_summary[reg], trend='n', order=(4, 0, 2), seasonal_order=(0, 1, 0, 30))
    results = mod.fit()
    print(results.summary())
    daily_summary['forecast'] = results.predict(start=1200, end=1369, dynamic=True)
    daily_summary['onestepforcast'] = results.predict(start=1200, end=1369, dynamic=False)
    daily_summary[[reg, 'forecast', 'onestepforcast']].plot(figsize=(12, 8))
    plt.title('SARIMA method')
    plt.savefig('Fig/forcast_season %s' % reg)
    plt.show()
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals \n SARIMA method", ax=ax[0])
    residuals.plot(kind='kde', title='Density \n SARIMA method', ax=ax[1])
    plt.savefig('Fig/residual_season %s' % reg)
    plt.show()
    ac1 = forecast_accuracy(daily_summary['forecast'][1200:1369], daily_summary[reg][1200:1369])
    ac2= forecast_accuracy(daily_summary['onestepforcast'][1200:1369], daily_summary[reg][1200:1369])
    print('%s season forecast' % reg)
    print(ac1)
    print('%s season onestepforcast' % reg)
    print(ac2)
#####

    mod = SARIMAX(monthly_summary[reg], trend='n', order=(4,1,2), seasonal_order=(0,0,0,12))
    results = mod.fit()
    print(results.summary())
    monthly_summary['forecast'] = results.predict(start = 30, end= 46, dynamic= True)
    monthly_summary['onestepforcast'] = results.predict(start = 30, end= 46, dynamic= False)
    monthly_summary[[reg, 'forecast','onestepforcast']].plot(figsize=(12, 8))
    plt.title('ARIMA method monthly')
    plt.savefig('Fig/forcast_nonseason_monthly %s' % reg)
    plt.show()
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals \n ARIMA method monthly", ax=ax[0])
    residuals.plot(kind='kde', title='Density \n ARIMA method monthly', ax=ax[1])
    plt.savefig('Fig/residual_nonseason_monthly %s' % reg)
    plt.show()
    ac1 = forecast_accuracy(monthly_summary['forecast'][30:46], monthly_summary[reg][30:46])
    ac2 = forecast_accuracy(monthly_summary['onestepforcast'][30:46], monthly_summary[reg][30:46])
    print('%s nonseason forecast monthly' % reg)
    print(ac1)
    print('%s nonseason onestepforcast monthly' % reg)
    print(ac2)

    mod = SARIMAX(monthly_summary[reg], trend='n', order=(4, 1, 2), seasonal_order=(0, 1, 0, 12))
    results = mod.fit()
    print(results.summary())
    monthly_summary['forecast'] = results.predict(start=30, end=46, dynamic=True)
    monthly_summary['onestepforcast'] = results.predict(start=30, end=46, dynamic=False)
    monthly_summary[[reg, 'forecast', 'onestepforcast']].plot(figsize=(12, 8))
    plt.title('SARIMA method monthly')
    plt.savefig('Fig/forcast_season_monthly %s' % reg)
    plt.show()
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title="Residuals \n SARIMA method monthly ", ax=ax[0])
    residuals.plot(kind='kde', title='Density \n SARIMA method monthly', ax=ax[1])
    plt.savefig('Fig/residual_season_monthly %s' % reg)
    plt.show()
    ac1 = forecast_accuracy(monthly_summary['forecast'][30:46], monthly_summary[reg][30:46])
    ac2= forecast_accuracy(monthly_summary['onestepforcast'][30:46], monthly_summary[reg][30:46])
    print('%s season forecast monthly' % reg)
    print(ac1)
    print('%s season onestepforcast monthly' % reg)
    print(ac2)
#



