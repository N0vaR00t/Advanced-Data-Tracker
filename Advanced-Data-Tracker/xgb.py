import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pandas.tseries.offsets import BDay
from data_preprocessing import DataPreProcessing
import talib as tb
 
def plot_predictions(predicted_prices, actual_prices, start_date):

    dates = [start_date + timedelta(days=i) for i in range(len(predicted_prices))]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, predicted_prices, label="Predicted Prices", marker='o', linestyle='dashed')
    plt.plot(dates[:len(actual_prices)], actual_prices, label="Actual Prices", marker='x', linestyle='solid')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction vs Actual")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def calculate_mape(actual, predicted):

    return np.mean(np.abs((actual - predicted) / actual)) * 100

def calculate_rmse(actual, predicted):

    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_dmi(df):

    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    close = df['Close'].values.flatten()

    plus_di = tb.PLUS_DI(high, low, close, timeperiod=14)
    minus_di = tb.MINUS_DI(high, low, close, timeperiod=14)
    
    return plus_di, minus_di

def dmi_directional_accuracy(predicted, plus_di, minus_di):
    correct = 0
    total = len(predicted) - 1

    for i in range(total):
        pred_direction = np.sign(predicted[i+1] - predicted[i])
        dmi_direction = 1 if plus_di[i] > minus_di[i] else -1

        if pred_direction == dmi_direction:
            correct += 1

    return (correct / total) * 100

data_prep = DataPreProcessing(seq_length=10, num_days_ahead=10)
data_prep.input()  
data_prep.fetch() 
data_prep.preprocess() 
data_prep.heatmap()
X_train, y_train = data_prep.X_train, data_prep.y_train
X_train_2d = X_train.reshape(X_train.shape[0], -1)
print(f"Shape of X_train after reshaping: {X_train_2d.shape}")

model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5)
model.fit(X_train_2d, y_train)
num_days = 10
today = datetime.today().date()
start_date = today - BDay(7 + num_days)  
end_date = start_date + BDay(num_days)  
predicted_prices = data_prep.multi_step_forecast(model, num_days=num_days)

print("\nPredicted Prices for the next 5 trading days:")

for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i} ({start_date + BDay(i-1)}): ${price:.2f}")

actual_prices = data_prep.fetch_actual_prices(start_date, end_date)
min_length = min(len(actual_prices), len(predicted_prices))
r2 = r2_score(actual_prices[:min_length], predicted_prices[:min_length])
print(f"R² Score: {r2:.4f}")
mape = calculate_mape(actual_prices[:min_length], predicted_prices[:min_length])
print(f"MAPE: {mape:.4f}%")
rmse = calculate_rmse(actual_prices[:min_length], predicted_prices[:min_length])
print(f"RMSE: {rmse:.4f}")
plus_di, minus_di = calculate_dmi(data_prep.df)
print(f"Max +DI: {np.max(plus_di[-min_length:]):.4f}")
print(f"Max -DI: {np.max(minus_di[-min_length:]):.4f}")
plus_di_recent = plus_di[-min_length:]
minus_di_recent = minus_di[-min_length:]
dmi_acc = dmi_directional_accuracy(predicted_prices[:min_length], plus_di_recent, minus_di_recent)
print(f"DMI Directional Accuracy: {dmi_acc:.2f}%")
plot_predictions(predicted_prices, actual_prices, start_date)
