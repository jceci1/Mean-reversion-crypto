import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# --- Load Data ---

# Replace with your actual filenames
eth_df = pd.read_csv('C:\\Users\\Joseph\\OneDrive\\Desktop\\Mean Reversion\\pairs data\\ETHUSDT_1h.csv', parse_dates=['open_time'])
btc_df = pd.read_csv('pairs data\\BTCUSDT_1h.csv', parse_dates=['open_time'])

# --- Prepare Data ---

# Rename columns for clarity
eth_df = eth_df[['open_time', 'close']].rename(columns={'close': 'eth_close'})
btc_df = btc_df[['open_time', 'close']].rename(columns={'close': 'btc_close'})

# Merge on timestamp
merged = pd.merge(eth_df, btc_df, on='open_time', how='inner')
merged.dropna(inplace=True)

# --- Log Prices ---
merged['log_eth'] = np.log(merged['eth_close'])
merged['log_btc'] = np.log(merged['btc_close'])

# --- Step 1: Engle-Granger Cointegration Test ---

# Regress ETH on BTC
X = sm.add_constant(merged['log_btc'])
model = sm.OLS(merged['log_eth'], X).fit()
merged['residual'] = model.resid

# Step 2: ADF test on residuals
adf_result = adfuller(merged['residual'])

print("=== Engle-Granger Cointegration Test ===")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"  {key}: {value}")

if adf_result[1] < 0.05:
    print("✅ Series are cointegrated (reject null of unit root in residuals)")
else:
    print("❌ Series are NOT cointegrated (cannot reject unit root in residuals)")

# --- Step 3: Hurst Exponent ---

def hurst_exponent(ts):
    lags = range(2, 100)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return 2.0 * poly[0]

hurst = hurst_exponent(merged['residual'])

print("\n=== Hurst Exponent of Spread (Residuals) ===")
print(f"Hurst Exponent: {hurst:.4f}")
if hurst < 0.5:
    print("Mean-reverting")
elif hurst > 0.5:
    print("Trending")
else:
    print("Random walk")

# Optional: plot residuals
plt.figure(figsize=(12, 4))
plt.plot(merged['open_time'], merged['residual'])
plt.title('Residuals from Cointegration (Spread)')
plt.xlabel('Time')
plt.ylabel('Residual (Spread)')
plt.grid(True)
plt.tight_layout()
plt.show()
