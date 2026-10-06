# stock-predictor-2-
# Ensemble Stock Direction Predictor

A Streamlit-based machine learning application that predicts the **next trading session's stock price direction** using an ensemble of machine learning models.

The application combines technical indicators, multiple classification models, model evaluation, and backtesting to provide an interactive stock market analysis dashboard.

## Features

-  Interactive stock market dashboard
-  Next-day direction prediction
-  Ensemble of:
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - Logistic Regression
-  Technical indicators including:
  - Moving Averages
  - RSI
  - MACD
  - Bollinger Bands
  - Stochastic Oscillator
  - Williams %R
  - ATR
  - ROC
  - Volatility
  - Volume changes
-  Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - Confusion Matrix
  - Statistical test against baseline
-  Walk-forward out-of-sample evaluation
-  Strategy backtesting against Buy & Hold
-  Sharpe Ratio and Maximum Drawdown
-  Random Forest feature importance
-  Interactive Plotly visualizations

## How It Works

1. The user enters a stock ticker such as `AAPL`, `TSLA`, or `RELIANCE.NS`.
2. Historical market data is retrieved from Stooq or Yahoo Finance.
3. Technical indicators and other features are calculated.
4. The data is divided chronologically into training and testing periods.
5. Multiple machine learning models are trained and tuned.
6. The models are combined using a soft-voting ensemble.
7. The ensemble predicts whether the next trading session is likely to move up or down.
8. The model is evaluated using out-of-sample testing and walk-forward validation.
9. A simple backtest compares the strategy with a Buy & Hold approach.

## Project Structure

```text
Ensemble-Stock-Direction-Predictor/
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Deployment

This application can be deployed using **Streamlit Community Cloud** by connecting the GitHub repository and selecting `app.py` as the main file.

## Important Note

This project is intended for **academic and experimental purposes**. Stock markets are highly unpredictable, and the predictions should not be considered financial advice.

The backtest is simplified and does not fully account for factors such as slippage, taxes, liquidity, and other real-world trading costs.
