import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet, ElasticNetCV, BayesianRidge, \
    OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from operator import itemgetter

df = pd.read_csv('./data/AAPL.csv', index_col='Date', parse_dates=True)

# print('\nPrint loaded data:\n', df.tail())

# print('\nNull rows count: ', df.isnull().T.any().T.sum())
# print('Null rows: ')
# print(df[df.isnull().T.any().T])

# Fill missing values using ffill method
df = df.ffill()

# print('\nNull rows count: ', df.isnull().T.any().T.sum())

# === Feature engineering ===

df_regularized = df.loc[:, ['Adj Close', 'Volume']]
# High-Low Percentage
df_regularized['HL Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.
# Percentage Change
df_regularized['Pct Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.

# Moving Avg
MOVING_AVERAGE_PERIOD = 200
MIN_PERIODS = 30
df_regularized['Moving Average'] = df_regularized['Adj Close'].rolling(MOVING_AVERAGE_PERIOD,
                                                                       min_periods=MIN_PERIODS).mean()
df_regularized.dropna(inplace=True)
# print('\nDF Regularized:\n', df_regularized.tail())

# df_regularized.filter(['HL Pct', 'Pct Change']).plot()
# plt.legend()
#
# df_regularized.filter(['Adj Close', 'Moving Average']).plot()
# plt.legend()
#
# plt.figure()
# df_regularized['Volume'].plot()
# plt.legend()
# plt.show()

DAYS_TO_FORECAST = 50  # Last N days to predict based on historical data
FORECASTING_COLUMN = 'Adj Close'
df_regularized['label'] = df_regularized[FORECASTING_COLUMN].shift(-1)

df_regularized_train = df_regularized.copy()
df_regularized_train.dropna(inplace=True)

X = np.array(df_regularized_train.drop(['label'], 'columns'))
# Rescale X for regression
X = scale(X)
X = X[:-DAYS_TO_FORECAST]
y = np.array(df_regularized_train['label'][:-DAYS_TO_FORECAST])

# Prepare data for forecasting
X_forecast = X[-DAYS_TO_FORECAST:]

# Split data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

models_summary = []


def evaluate_model(model_name, model):
    model.fit(X_train, y_train)

    # Get confidence
    model_confidence = model.score(X_test, y_test)

    # Get forecast
    y_forecast = model.predict(X_forecast)

    # Assign default NaN predictions
    model_df_regularized_forecast = df_regularized.copy()
    model_df_regularized_forecast['Forecast'] = np.nan

    for idx, y_f in enumerate(y_forecast):
        model_df_regularized_forecast.iloc[-DAYS_TO_FORECAST + idx] = list(
            model_df_regularized_forecast.drop('Forecast', 'columns').iloc[-DAYS_TO_FORECAST + idx].values) + [y_f]

    # Add last n entries - forecast rows
    expected_data = model_df_regularized_forecast.iloc[-DAYS_TO_FORECAST:]['Adj Close']
    forecast_data = model_df_regularized_forecast.iloc[-DAYS_TO_FORECAST:]['Forecast']

    model_summary = {
        'confidence': model_confidence * 100,
        'expected_values': expected_data,
        'forecast_values': forecast_data,
        'mae': mean_absolute_error(expected_data, forecast_data),
        'mse': mean_squared_error(expected_data, forecast_data),
        'rmse': np.sqrt(mean_absolute_error(expected_data, forecast_data)),
        'name': model_name
    }

    return model_summary


# Hyperparams
max_iter = int(1e3)
alpha = .1
alphas = [1e-4, 1e-3, 1e-2, 1e-1]

models_summary.append(evaluate_model('Linear', LinearRegression()))

models_summary.append(evaluate_model('KNN', KNeighborsRegressor(n_neighbors=2)))

models_summary.append(evaluate_model('Lasso', Lasso(alpha=alpha, max_iter=max_iter)))

models_summary.append(evaluate_model('Ridge', Ridge(alpha=alpha, max_iter=max_iter)))

models_summary.append(evaluate_model('Ridge CV', RidgeCV(alphas=alphas)))

models_summary.append(evaluate_model('Kernel Ridge', KernelRidge(alpha=alpha)))

models_summary.append(evaluate_model('Elastic Net', ElasticNet(alpha=alpha, max_iter=max_iter)))

models_summary.append(evaluate_model('Elastic Net CV', ElasticNetCV(alphas=alphas, max_iter=max_iter)))

models_summary.append(evaluate_model('Bayesian Ridge', BayesianRidge(n_iter=max_iter)))

models_summary.append(evaluate_model('Orthogonal Matching Pursuit', OrthogonalMatchingPursuit()))

models_summary.append(evaluate_model('Orthogonal Matching Pursuit CV', OrthogonalMatchingPursuitCV()))

print('Models sorted by confidence')
for model_summary in sorted(models_summary, key=itemgetter('confidence'), reverse=True):
    print('| {} | {}% | {} | {} | {} |'.format(
        model_summary['name'],
        round(model_summary['confidence'], 4),
        round(model_summary['mae'], 3),
        round(model_summary['mse'], 3),
        round(model_summary['rmse'], 3),
    ))

print('Models sorted by RSME')
for model_summary in sorted(models_summary, key=itemgetter('rmse')):
    print('| {} | {}% | {} | {} | {} |'.format(
        model_summary['name'],
        round(model_summary['confidence'], 4),
        round(model_summary['mae'], 3),
        round(model_summary['mse'], 3),
        round(model_summary['rmse'], 6),
    ))

    plt.figure()
    plt.title(model_summary['name'])
    model_summary['expected_values'].plot()
    model_summary['forecast_values'].plot()
    plt.savefig('./assets/{}.jpg'.format(model_summary['name']))

plt.show()
