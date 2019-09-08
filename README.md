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
| KNN    | 97.162% | 9.581 | 137.833 | 11.74 |
| Linear | 97.557% | 19.318 | 472.715 | 21.742 |
| Lasso  | 96.156% | 19.406 | 476.549 | 21.83 |

In the next three paragraphs you can find predictions using KNN, Linear, and Lasso regression models.

### KNN regression

Settings:
- Neighbors: 4

![knn-regresssion](./assets/KNN%20regression.png)

### Linear regression

![linear-regresssion](./assets/Linear%20Regression.png)

### Lasso regression

Settings:
- Alpha: 0.05

![lasso-regression](./assets/Lasso%20regression.png)

