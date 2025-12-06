import pandas as pd 
import numpy as np
# Importation of data 
#Attention plusieurs stocks
data = pd.read_csv('data/all_stocks_5yr.csv')



#On trasnforme close_prices en DataFrame et on met x colomne pour le nombre de stocks
close_prices = data.pivot(index='date', columns='Name', values='close')
returns = np.log(close_prices / close_prices.shift(1))
print(returns.shape)


returns_daily = returns.mean()
cov_daily = returns.cov()
print(returns_daily.shape)
print(cov_daily.shape)


