import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/AAPL.csv', index_col='Date', parse_dates=True)

print('\nPrint loaded data:\n', df.tail())

print('\nNull rows count: ', df.isnull().T.any().T.sum())
print('Null rows: ')
print(df[df.isnull().T.any().T])

# Fill missing values using ffill method
df = df.ffill()

print('\nNull rows count: ', df.isnull().T.any().T.sum())

# Feature engineering
df_regularized = df.loc[:, ['Adj Close', 'Volume']]
# High-Low Percentage
df_regularized['HL Pct'] = (df['High'] - df['Low']) / df['Close'] * 100.
# Percentage Change
df_regularized['Pct Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.
print('\nDF Regularized:\n', df_regularized.tail())

# Data for forecast:
forecast_rows = math.ceil(0.01 * len(df_regularized))
print('\nTest rows: ', forecast_rows, ' out of: ', len(df_regularized))

FORECASTING_COLUMN = 'Adj Close'
df_regularized['label'] = df_regularized[FORECASTING_COLUMN].shift(-forecast_rows)

# Remove empty values after shifting the column data
df_regularized_forecast = df_regularized.copy()

# Display the head
print('df_regularized:\n', df_regularized.head())

# drop - returns all columns except specified ones in the list
X = np.array(df_regularized.drop(['label'], 'columns'))
# Rescale X for regression
X = scale(X)
X = X[:-forecast_rows]
X_forecast = X[-forecast_rows:]

df_regularized.dropna(inplace=True)

# Assign labels
y = np.array(df_regularized['label'])

# Display first 10 Xs (independent variable) and ys (dependent variable)
print('X: ', X[:10])
print('y: ', y[:10])
print('df_regularized:\n', df_regularized.head())

# Split data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Verify the data
print('X train: ', X_train.shape, X_train[:10])
print('is X train finite: ', np.isfinite(X_train).all())
print('is y train finite: ', np.isfinite(y_train).all())

#
# ==== Linear Regression ====
#
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Display coefficients
print('Coef: ', linear_regression_model.coef_)

# Display confidence
linear_regression_confidence = linear_regression_model.score(X_test, y_test)

# Prediction
y_forecast = linear_regression_model.predict(X_forecast)

# Assign default NaN predictions
lr_df_regularized_forecast = df_regularized_forecast.copy()
lr_df_regularized_forecast['Forecast'] = np.nan

for idx, y_f in enumerate(y_forecast):
    lr_df_regularized_forecast.iloc[-forecast_rows + idx] = list(
        lr_df_regularized_forecast.drop('Forecast', 'columns').iloc[-forecast_rows + idx].values) + [y_f]

# Display last 200 entries
plt.figure()
lr_df_regularized_forecast.iloc[-200:]['Adj Close'].plot()
lr_df_regularized_forecast.iloc[-200:]['Forecast'].plot()
plt.title('Linear Regression')
# plt.show()

#
# ==== KNN Regression ====
#

knn_model = KNeighborsRegressor(n_neighbors=4)
knn_model.fit(X_train, y_train)

# Display confidence
knn_confidence = knn_model.score(X_test, y_test)

# Prediction
y_forecast = knn_model.predict(X_forecast)

# Assign default NaN predictions
knn_df_regularized_forecast = df_regularized_forecast.copy()
knn_df_regularized_forecast['Forecast'] = np.nan

for idx, y_f in enumerate(y_forecast):
    knn_df_regularized_forecast.iloc[-forecast_rows + idx] = list(
        knn_df_regularized_forecast.drop('Forecast', 'columns').iloc[-forecast_rows + idx].values) + [y_f]

# Display last 200 entries
plt.figure()
knn_df_regularized_forecast.iloc[-200:]['Adj Close'].plot()
knn_df_regularized_forecast.iloc[-200:]['Forecast'].plot()
plt.title('KNN Regression')

#
# ==== Lasso Regression ====
#

lasso_model = Lasso(alpha=0.05)
lasso_model.fit(X_train, y_train)

# Display confidence
lasso_confidence = lasso_model.score(X_test, y_test)

# Prediction
y_forecast = lasso_model.predict(X_forecast)

# Assign default NaN predictions
lasso_df_regularized_forecast = df_regularized_forecast.copy()
lasso_df_regularized_forecast['Forecast'] = np.nan

for idx, y_f in enumerate(y_forecast):
    lasso_df_regularized_forecast.iloc[-forecast_rows + idx] = list(
        lasso_df_regularized_forecast.drop('Forecast', 'columns').iloc[-forecast_rows + idx].values) + [y_f]

# Display last 200 entries
plt.figure()
lasso_df_regularized_forecast.iloc[-200:]['Adj Close'].plot()
lasso_df_regularized_forecast.iloc[-200:]['Forecast'].plot()
plt.title('Lasso Regression')
plt.show()

print('Confidences: ')
print('- Linear regression: ', round(linear_regression_confidence, 5))
print('- KNN regression: ', round(knn_confidence, 5))
print('- Lasso regression: ', round(lasso_confidence, 5))

# Confidences:
# - Linear regression: 0.97065
# - KNN regression:    0.98265
# - Lasso regression:  0.97064
