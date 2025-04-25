import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# --- User Parameters ---
TICKER = 'AAPL'  # Change to any stock symbol you want
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
PREDICT_DAYS = 30  # Number of days to forecast

# --- 1. Download Data ---
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

# --- 2. Preprocess ---
prices = data['Close'].dropna()

# --- 3. Train/Test Split ---
train = prices[:-PREDICT_DAYS]
test = prices[-PREDICT_DAYS:]

# --- 4. Fit ARIMA Model ---
# (p,d,q) can be tuned; (5,1,0) is a safe simple start
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# --- 5. Forecast ---
forecast = model_fit.forecast(steps=PREDICT_DAYS)

# --- 6. Plot Results ---
plt.figure(figsize=(10,5))
plt.plot(prices.index, prices, label='Actual')
plt.plot(test.index, forecast, label='Predicted', linestyle='--')
plt.title(f'{TICKER} Stock Price Prediction (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# --- 6b. Show Actual vs Predicted Table ---
comparison_df = pd.DataFrame({
    'Date': test.index.strftime('%Y-%m-%d'),
    'Actual Price': test.values,
    'Predicted Price': forecast.values
})
print('\nActual vs Predicted Prices:')
print(comparison_df.to_string(index=False))

# --- 7. Print RMSE ---
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.2f}')
