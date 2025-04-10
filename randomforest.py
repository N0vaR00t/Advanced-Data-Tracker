import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from data_preprocessing import DataPreProcessing 

def plot_predictions(predicted_prices, actual_prices, start_date):
    """Plot predicted vs actual closing prices."""
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

data_prep = DataPreProcessing(seq_length=10, num_days_ahead=5)
data_prep.input() 
data_prep.fetch() 
data_prep.preprocess()

X_train, y_train = data_prep.X_train, data_prep.y_train
X_train_2d = X_train.reshape(X_train.shape[0], -1)

print(f"Shape of X_train after reshaping: {X_train_2d.shape}")

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_2d, y_train)

num_days = 5
today = datetime.today().date()
start_date = today - BDay(7 + num_days) 
end_date = start_date + BDay(num_days)  
predicted_prices = data_prep.multi_step_forecast(model, num_days=num_days)

print("\nPredicted Prices for the next 10 trading days:")

for i, price in enumerate(predicted_prices, start=1):
    print(f"Day {i} ({start_date + BDay(i-1)}): ${price:.2f}")

actual_prices = data_prep.fetch_actual_prices(start_date, end_date)
min_length = min(len(actual_prices), len(predicted_prices))
r2 = r2_score(actual_prices[:min_length], predicted_prices[:min_length])

print(f"R² Score: {r2:.4f}")
plot_predictions(predicted_prices, actual_prices, start_date)
