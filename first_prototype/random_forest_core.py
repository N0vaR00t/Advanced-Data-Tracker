import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("all imported")

df = yf.download("AAPL", start="2010-11-01", end="2020-11-01")               # data using yfinance
df["Diff"] = df.Close.diff()
df["SMA_2"] = df.Close.rolling(2).mean()
df["Force_Index"] = df["Close"] * df["Volume"]
df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
df = df.drop(
    ["Open", "High", "Low", "Close", "Volume", "Diff", "Adj Close"],
    axis=1,
).dropna()

print(df)

X = df.drop(["y"], axis=1).values                                             # to prepare data for training
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)                     # to train Random Forest
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")                                                  # accuracy score

