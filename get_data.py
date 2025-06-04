import os
import time
import requests
import pandas as pd

#fetch data from Binance Rest API
def get_klines(symbol, interval, start_time=None, end_time=None, limit=1000, retries=3):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    for attempt in range(retries):
        #combats problems with vpn
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"error fetching {symbol} ({interval}) attempt {attempt + 1}/{retries}: {e}")
            time.sleep(0.5)

    print(f"failed to fetch {symbol} ({interval}) after {retries} retries.")
    return []


#formats and sets up Binance data collection
def get_data(candles_needed, candle_size, symbol, interval):
    
    now = int(time.time() * 1000)
    candles = []
    start_time = now - candles_needed*candle_size

    #Loops as needed since Binance only allows 1000 candles per API call
    while candles_needed > 0:
        fetch_number = min(candles_needed, 1000)

        data = get_klines(symbol, interval, start_time=start_time, limit=fetch_number)

        if not data:
            break

        candles.extend(data)
        start_time = data[-1][0] + candle_size
        candles_needed -= len(data)

        #avoid rate limiting
        time.sleep(0.5)

    df = pd.DataFrame(candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close"] = df["close"].astype(float)
    df.set_index("open_time", inplace=True)
    return df[["close"]]


#saves as csv
def save_to_csv(df, symbol, interval, folder="crypto_data"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{symbol}_{interval}.csv"
    path = os.path.join(folder, filename)
    df.to_csv(path)
    print(f"Saved {symbol} data to {path}")


#token list
tokens = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT',
    'ARBUSDT', 'OPUSDT', 'MATICUSDT',
    'LDOUSDT', 
    'UNIUSDT', 'AAVEUSDT', 'CRVUSDT', 'COMPUSDT', 'DYDXUSDT', 'GMXUSDT', 'SUSHIUSDT',
    'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT', 'TRUMPUSDT', 
    'FETUSDT', 'INJUSDT',
    'LUNAUSDT', 'FTTUSDT'
]


#candle sizes in milliseconds
ticks_15m = 60*15*1000
ticks_1h = ticks_15m*4
ticks_4h = ticks_1h*4
ticks_1d =  ticks_1h*24


#gets data for all tokens at all lengths
for token in tokens:
    df_15m = get_data(4500, ticks_15m, token, '15m')
    df_1h = get_data(2000, ticks_1h, token, '1h')
    df_4h = get_data(1200, ticks_4h, token, '4h')
    df_1d = get_data(500, ticks_1d, token, '1d')
    save_to_csv(df_15m, token, "15m", folder="pairs data")
    save_to_csv(df_1h, token, "1h", folder="pairs data")
    save_to_csv(df_4h, token, "4h", folder="pairs data")
    save_to_csv(df_1d, token, "1d", folder="pairs data")



