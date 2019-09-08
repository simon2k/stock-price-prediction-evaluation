# Stock Price Prediction Evaluation

This project is indented to present a small evaluation of different types of regression models for predicting stock prices for [AAPL](https://finance.yahoo.com/quote/AAPL/history?p=AAPL). It is focused on a simple linear, KNN and lasso regression models.

## Setup

Steps:

- Download the repository
- Install packages: `pip install -r requirements.txt`
- Run: `python comparison.py`

## Evaluation

### Comparison

| Method | Confidence | Mean Absolute Error | Mean Squared Error | Root Mean Squared Error |
| :---: | ---: | ---: | ---: | ---: |
| KNN    | 98.372% |  3.602 |  21.156 |  4.6 |
| Linear | 97.116% |  9.241 |  114.052 |  10.68 |
| Lasso  | 97.124% |  9.544 |  121.564 |  11.026 |

#### Moving Average for Adj Close

![moving-average-for-adj-close](./assets/moving-avergage-for-adj-close.png)

In the next three paragraphs you can find predictions using KNN, Linear, and Lasso regression models.

### KNN regression

Settings:
- Neighbors: 2

![knn-regresssion](./assets/KNN%20regression.png)

### Linear regression

![linear-regresssion](./assets/Linear%20Regression.png)

### Lasso regression

Settings:
- Alpha: 0.1

![lasso-regression](./assets/Lasso%20regression.png)

