"""
Uses moving averages, relative strength, and volatility metrics to categorize each bar as one of three states: upward trending, downward trending, or flat. 

Upward trending is defined as a the price is generally rising over a period of greater than 15 days. 

Downward trending is defined as a the price is generally decreasing over a period of greater than 15 days. 

Flat is defined as a the price is generally remaining within a stable range over a period of greater than 15 days.
"""

import pandas as pd
import numpy as np

# Load data and preprocess
df = pd.read_csv('ohlc_data.csv')
df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Calculate Moving Average (MA30)
df['MA30'] = df['close'].rolling(window=30).mean()

# Calculate True Range (TR) and Average True Range (ATR30)
df['prev_close'] = df['close'].shift(1)
df['TR'] = np.maximum(
    df['high'] - df['low'],
    np.abs(df['high'] - df['prev_close']),
    np.abs(df['low'] - df['prev_close'])
)
df['ATR30'] = df['TR'].rolling(window=30).mean()

# Calculate Slope of MA30 (15-day difference)
df['MA30_15days_ago'] = df['MA30'].shift(15)
df['MA30_slope'] = df['MA30'] - df['MA30_15days_ago']

# Calculate RSI (14-day period)
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Calculate Bollinger Bands
df['MA20'] = df['close'].rolling(window=20).mean()
df['std20'] = df['close'].rolling(window=20).std()
df['UpperBand'] = df['MA20'] + 2 * df['std20']
df['LowerBand'] = df['MA20'] - 2 * df['std20']

# Define volatility threshold
k = 0.5
df['volatility_threshold'] = k * df['ATR30']

# Classify states
conditions = [
    (df['MA30_slope'] > df['volatility_threshold']) & 
    (df['close'] > df['MA30']) & 
    (df['RSI'] > 50),
    
    (df['MA30_slope'] < -df['volatility_threshold']) & 
    (df['close'] < df['MA30']) & 
    (df['RSI'] < 50),
    
    (df['MA30_slope'].abs() <= df['volatility_threshold']) & 
    (df['close'] >= df['LowerBand']) & 
    (df['close'] <= df['UpperBand']) & 
    (df['RSI'].between(40, 60))
]
choices = ['up', 'down', 'flat']
df['state'] = np.select(conditions, choices, default='flat')
df.loc[df['MA30_slope'].isna(), 'state'] = np.nan  # Handle initial NaNs

# Output results
result = df[['timestamp', 'close', 'state']]
print(result)
