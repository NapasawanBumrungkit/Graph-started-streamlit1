import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


df = pd.read_csv("TSLA_2010-06-29_2025-02-13 (1).csv")


print(df.head())
print(df.info())


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


plt.figure(figsize=(10, 6))
plt.plot(df['Close'])
plt.title('Stock Price of TSLA Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()


df = df.dropna(subset=['Close'])  
X = df[['Open', 'High', 'Low', 'Volume']]  
y = df['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

xg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xg_model.fit(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

mse_lr, rmse_lr = evaluate_model(y_test, y_pred_lr)
mse_rf, rmse_rf = evaluate_model(y_test, y_pred_rf)
mse_xg, rmse_xg = evaluate_model(y_test, y_pred_xg)

print(f"Linear Regression - MSE: {mse_lr}, RMSE: {rmse_lr}")
print(f"Random Forest - MSE: {mse_rf}, RMSE: {rmse_rf}")
print(f"XGBoost - MSE: {mse_xg}, RMSE: {rmse_xg}")

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Prices', color='blue')
plt.plot(y_test.index, y_pred_lr, label='Linear Regression', color='red', linestyle='--')
plt.plot(y_test.index, y_pred_rf, label='Random Forest', color='green', linestyle='--')
plt.plot(y_test.index, y_pred_xg, label='XGBoost', color='orange', linestyle='--')
plt.legend()
plt.title('Stock Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
