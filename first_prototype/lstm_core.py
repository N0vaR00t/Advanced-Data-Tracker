import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("all imported")

df = yf.download("AAPL", start="2010-11-01", end="2020-11-01")                              # data using yfinance
df["Diff"] = df.Close.diff()
df["SMA_2"] = df.Close.rolling(2).mean()
df["Force_Index"] = df["Close"] * df["Volume"]
df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
df = df.drop(
    ["Open", "High", "Low", "Close", "Volume", "Diff", "Adj Close"],
    axis=1,
).dropna()

scaler = MinMaxScaler()                                                                             # to prepare data
X = scaler.fit_transform(df.drop("y", axis=1))
y = df["y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)     # to split train-test
                                                                                                    # to generate time series data
n_input = 15                                                                                        # window size
train_gen = TimeseriesGenerator(X_train, y_train, length=n_input, batch_size=32)
test_gen = TimeseriesGenerator(X_test, y_test, length=n_input, batch_size=32)

model = Sequential([                                                                                # to build LSTM model
    LSTM(50, activation='relu', input_shape=(n_input, X_train.shape[1])),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=10, verbose=1)                                                          # to train model


y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = y_test[n_input:]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")                                            # accuracy score


