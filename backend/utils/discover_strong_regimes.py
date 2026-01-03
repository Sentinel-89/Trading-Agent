import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("backend/data/processed/TCS_labeled.csv")

# Parse date
df["Date"] = pd.to_datetime(df["Date"])

# Plot price
plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.title("RELIANCE — Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

print("Index of chosen start date is", df[df["Date"] == "2021-07-29"].index)
