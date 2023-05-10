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

print(df.head())

plt.figure(figsize = (15,6))
plt.plot(df, lw=5, color="r")
plt.title("1st graph", fontsize = 20)
plt.show()
#Si tiene tendencia y es cíclico ya que se observan cambios repetitivos con una temporalidad amplia.

result = adfuller(df.traffic.dropna())
print('ADF Statistic %f' % result [0])
print('p-value; %f' % result[1])
#No es estacionario p-value mayor a 5%, se requiere hacer diferenciación


plt.rcParams.update({'figure.figsize':(12,15), 'figure.dpi':120})
#Serie original
fig, axes = plt.subplots(3, 2, sharex=False)
axes[0,0].plot(df.traffic, color = "r"); axes[0,0].set_title('Serie original')
plot_acf(df.traffic, ax=axes[0,1], color = "r")

#Primera diferenciación
axes[1,0].plot(df.traffic.diff(), color = "g"); axes[1,0].set_title('Diferenciación de primer orden')
plot_acf(df.traffic.diff().dropna(), ax=axes[1,1], color = "g")

#Segunda diferenciación
axes[2,0].plot(df.traffic.diff().diff(), color = "b"); axes[2,0].set_title('Diferenciación de segundo orden')
plot_acf(df.traffic.diff().diff().dropna(), ax=axes[2,1], color = "b")
plt.show()

#Primera difrenciación
df['traffic_diff'] = df.traffic.diff()
df = df.dropna()  
result = adfuller(df.traffic_diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


#Segunda difrenciación
df['traffic_diff'] = df.traffic.diff().diff()
df = df.dropna()  
result = adfuller(df.traffic_diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

#Nos quedamos con la segunda diferenciación, ya que tiene un p-value menor. 
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi': 120})

#Gráfica PACF de la primera serie diferenciada
fig, axes = plt.subplots(1,2, sharex=False)
axes[0].plot(df.traffic.diff().diff(), color  = "orangered"); axes[0].set_title("Pimera diferenciación")
axes[1].set(ylim=(-1,4))
plot_pacf(df.traffic.diff().dropna(), ax = axes[1], color = "orangered")
plt.show()
#Podemos observar que el retraso del primer PACF es muy significativo ya que sobrepasa el intervalo de confianza por mucho, el segundo también cuenta con un comportamiento similar. Si actuamos de manera conservadora, fijamos el valor de p como 1 tentativamente.


fig, axes = plt.subplots(1,2, sharex=False)
axes[0].plot(df.traffic.diff().diff(), color  = "fuchsia"); axes[0].set_title("Pimera diferenciación")
axes[1].set(ylim=(-1,4))
plot_acf(df.traffic.diff().dropna(), ax = axes[1], color = "fuchsia")
plt.title("Autocorrelación simple")
plt.show()


#Se utilizó un bucle anidado para encontrar los valores precisos, los puede visualizar en el otro documento llamado 'p_d_q_for.py', el rango
#se puede ampliar tanto como se desee pero se corre el riesgo de que el modelo se sobreajuste.

endog = df['traffic']

model = ARIMA(endog, order=(3,1,3))
model_fit = model.fit()

# Mostrar los resultados del modelo
print(model_fit.summary())
print(sm.stats.acorr_ljungbox(model_fit.resid, lags=[10], return_df=True))
#El p value es igual a 0.981033 por lo tanto si hay presencia de ruido blanco.

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
#Graficar errores residuales
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuales", ax=ax[0], color = "crimson", lw=2)
residuals.plot(kind="kde", title ="Densidad", ax=ax[1], color = "crimson", lw=2)
plt.show()


X = df[["traffic"]]
train, test = X[0:-15], X[-15:0]
model2 = ARIMA(train, order = (3,1,3))
fitted = model2.fit()


forecast_values, standard_error, confidence_interval = fitted.forecast(15, alpha=0.05)
fc_series = pd.Series(forecast_values, test.index)
lower_series = pd.Series(confidence_interval[:,0], test.index)
upper_series = pd.Series(confidence_interval[:,1], test.index)

plt.figure(figsize=(12,5))
plt.plot(train, label = "training", lw =2)
plt.plot(test, label = "actual", lw =2)
plt.plot(fc_series, label = "forecast", lw=2)
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color = "k", alpha = .15)
plt.title("Forecast vs Actuals")
plt.legend(loc = "upper left", fontsize = 10)
plt.show()


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred / y_true)) * 100
    print('Resultado de las métricas:-')
    print(f'MSE es : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE es : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE es : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE es : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 es : {metrics.r2_score(y_true, y_pred)}', end = '\n\n')

timeseries_evaluation_metrics_func(forecast_values, test)

#MSE es 40.61
#MAE es 4.54
#RMSE es 6.37
#MAPE es 12.33
#R2 es 0.91

#Las métricas resultadas son bastante buenas, sin embargo el hacer cambio podría generar un sobreajuste del modelo y no obtener los resultados deseados, además de que podemos concluir que el Modelo ARIMA no fue el mejor para esta serie de tiempo. 
























