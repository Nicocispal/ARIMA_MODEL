import itertools
import pandas as pd 
import numpy as np
import warnings
import itertools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf 
import pmdarima as pm
import statsmodels.api as sm
from numpy import log
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from arima_range_testing import *
warnings.filterwarnings('ignore')


df = 'downloads/website_data.csv'
df = pd.read_csv(df, encoding = 'utf8', sep = ',', on_bad_lines='warn')
def tested_range(): 
    p = range(0, 4)
    q = range(0, 4)
    d = range(0, 4)

    pdq = list(itertools.product(p, d, q))

    for param in pdq:
        try:
            model = ARIMA(df.traffic, order=param)
            results = model.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue
    return tested_range

