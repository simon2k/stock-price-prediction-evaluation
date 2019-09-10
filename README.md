# Stock Price Prediction Evaluation

This project is indented to present a small evaluation of different types of regression models for predicting stock prices for [AAPL](https://finance.yahoo.com/quote/AAPL/history?p=AAPL).

## Setup

Steps:

- Download the repository
- Install packages: `pip install -r requirements.txt`
- Run: `python comparison.py`

## Evaluation

### Comparison

| Method | Confidence | Mean Absolute Error | Mean Squared Error | Root Mean Squared Error |
| :---: | ---: | ---: | ---: | ---: |
| Linear | 96.54% | 9.267 | 114.54 | 3.044 |
| KNN | 98.39% | 3.717 | 23.151 | 1.928 |
| Lasso | 96.55% | 9.567 | 122.002 | 3.093 |
| Ridge | 96.54% | 9.261 | 114.403 | 3.043 |
| Ridge CV | 96.54% | 9.261 | 114.403 | 3.043 |
| Kernel Ridge | 60.8% | 41.406 | 1762.293 | 6.435 |
| Elastic Net | 96.47% | 8.639 | 100.439 | 2.939 |
| Elastic Net CV | 96.54% | 9.04 | 109.289 | 3.007 |
| Bayesian Ridge | 96.54% | 9.253 | 114.217 | 3.042 |
| Orthogonal Matching Pursuit | 96.4% | 18.776 | 448.374 | 4.333 |
| Orthogonal Matching Pursuit CV | 96.54% | 9.268 | 114.556 | 3.044 |

#### Moving Average for Adj Close

![moving-average-for-adj-close](./assets/moving-avergage-for-adj-close.png)

In the next paragraphs you can find predictions using selected methods.

Settings:
- Alpha: 0.1
- Number of iterations: 1000
- Alphas search: 0.0001, 0.0001, 0.001, 0.1
- Number of neighbors: 2

![](./assets/Bayesian%20Ridge.jpg)

![](./assets/Elastic%20Net.jpg)

![](./assets/Elastic%20Net%20CV.jpg)

![](./assets/Kernel%20Ridge.jpg)

![](./assets/KNN.jpg)

![](./assets/Lasso.jpg)

![](./assets/Linear.jpg)

![](./assets/Orthogonal%20Matching%20Pursuit.jpg)

![](./assets/Orthogonal%20Matching%20Pursuit%20CV.jpg)

![](./assets/Ridge.jpg)

![](./assets/Ridge%20CV.jpg)

