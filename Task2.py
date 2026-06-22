import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Download Apple stock data
df = yf.download("AAPL", start="2022-01-01", end="2026-01-01")

print("First 5 Rows")
print(df.head())

# Features
X = df[['Open', 'High', 'Low', 'Volume']]

# Target
y = df['Close']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# Graph
plt.figure(figsize=(10,6))

plt.plot(
    y_test.values[:100],
    label="Actual Price"
)

plt.plot(
    y_pred[:100],
    label="Predicted Price"
)

plt.legend()

plt.title("Actual vs Predicted Stock Prices")

plt.savefig("stock_prediction.png")

plt.show()

print("\nGraph Saved Successfully") 