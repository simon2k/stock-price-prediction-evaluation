# Stock Price Prediction Evaluation

This project is indented to present a small evaluation of different types of regression models for predicting stock prices for [AAPL](https://finance.yahoo.com/quote/AAPL/history?p=AAPL). It is focused on a simple linear, KNN and lasso regression models.

## Setup

Steps:

- Download the repository
- Install packages: `pip install -r requirements.txt`
- Run: `python comparison.py`

## Evaluation

In the next three paragraphs is placed a comparison between Linear, KNN and Lasso regression.

### Linear regression

![linear-regresssion](./assets/Linear%20Regression.png)

### KNN regression

Settings:
- Neighbors: 4

![knn-regresssion](./assets/KNN%20regression.png)

### Lasso regression

Settings:
- Alpha: 0.05

![lasso-regression](./assets/Lasso%20regression.png)

## Confidences comparison

- Linear regression: 0.97065
- KNN regression:    0.98265
- Lasso regression:  0.97064
