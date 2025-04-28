import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import talib as tb 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import BDay 

class DataPreProcessing:
    def __init__(self, seq_length=10, num_days_ahead=10, scaler_type='minmax'):
        self.df = None
        self.symbol = None
        self.seq_length = seq_length
        self.num_days_ahead = num_days_ahead 
        self.scaler_type = scaler_type
        self.scaler = self._initialize_scaler(scaler_type)
        self.X_train = None
        self.y_train = None

    def _initialize_scaler(self, scaler_type):

        if scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def input(self):

        self.symbol = input("Please enter a stock's symbol: ").upper()

    def fetch(self, start="2020-01-01", end="2025-04-02"):

        if end is None:
            end = datetime.today().strftime('%Y-%m-%d')
            
        self.df = yf.download(self.symbol, start=start, end=end, progress=False)

        if self.df.empty:
            raise ValueError("Error: No data retrieved. Check the stock symbol or date range.")

        print("Stock Data Preview:")
        print(self.df.head())

    def preprocess(self):

        if self.df is None or 'Close' not in self.df.columns:
            raise ValueError("Error: Data not loaded properly. Ensure 'Close' column exists.")

        self.df.dropna(inplace=True)

        c, h, l, v = [self.df[col].values.astype(np.float64) for col in ['Close', 'High', 'Low', 'Volume']]

        c = c.flatten()
        h = h.flatten()
        l = l.flatten()
        v = v.flatten()

        self.df['MA5'] = tb.MA(c, timeperiod=5)
        self.df['MA20'] = tb.MA(c, timeperiod=20)
        self.df['RSI'] = tb.RSI(c, timeperiod=14)
        self.df['MACD'], self.df['MACD_signal'], _ = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['ATR'] = tb.ATR(h, l, c, timeperiod=14)
        self.df['BOLLINGER_UPPER'], self.df['BOLLINGER_MIDDLE'], self.df['BOLLINGER_LOWER'] = tb.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.df['STOCH'], self.df['STOCH_SIGNAL'] = tb.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
        self.df['OBV'] = tb.OBV(c, v)
        self.df['MFI'] = tb.MFI(h, l, c, v, timeperiod=14)

        self.df.dropna(inplace=True)
        features = ['Close', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'ATR', 'BOLLINGER_UPPER', 'BOLLINGER_MIDDLE', 'BOLLINGER_LOWER', 'STOCH', 'STOCH_SIGNAL', 'OBV', 'MFI']
        self.df[features] = self.scaler.fit_transform(self.df[features])
        X, y = [], []
        scaled_data = self.df[features].values

        for i in range(self.seq_length, len(scaled_data) - self.num_days_ahead):

            X.append(scaled_data[i-self.seq_length:i]) 
            y.append(scaled_data[i:i+self.num_days_ahead, 0]) 

        self.X_train = np.array(X)
        self.y_train = np.array(y)

        print(f"Training Data Prepared: X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")

    def heatmap(self):

        corr_features = ['Close', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'ATR',
                         'BOLLINGER_UPPER', 'BOLLINGER_MIDDLE', 'BOLLINGER_LOWER',
                         'STOCH', 'STOCH_SIGNAL', 'OBV', 'MFI']
        
        corr_matrix = self.df[corr_features].corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Features (Including Close)')
        plt.tight_layout()
        plt.show()

    def multi_step_forecast(self, model, num_days=10):
    
        last_sequence = self.X_train[-1].reshape(1, -1)
        predicted = model.predict(last_sequence)[0] 
        predicted_reshaped = predicted.reshape(-1, 1)
        dummy_features = np.zeros((len(predicted), 15)) 
        predicted_data = np.hstack((predicted_reshaped, dummy_features))
        predicted_prices = self.scaler.inverse_transform(predicted_data)[:, 0] 

        return predicted_prices

    def multi_step_forecast_lstm(self, model, num_days=10):

        last_sequence = self.X_train[-1].reshape(1, self.seq_length, self.X_train.shape[2])  
        print("Shape of last_sequence:", last_sequence.shape) 
        predicted = model.predict(last_sequence, batch_size=1)[0] 
        predicted_reshaped = predicted.reshape(-1, 1)
        dummy_features = np.zeros((len(predicted), 15))
        predicted_data = np.hstack((predicted_reshaped, dummy_features))
        predicted_prices = self.scaler.inverse_transform(predicted_data)[:, 0]

        return predicted_prices

    def fetch_actual_prices(self, start_date, end_date):

        actual_df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)

        if actual_df.empty:
            raise ValueError("Error: No actual data available for comparison.")

        actual_prices = actual_df['Close'].values
        return actual_prices
