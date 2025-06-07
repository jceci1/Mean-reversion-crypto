import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import os
import itertools

def hurst_exponent(ts):
    """
    Calculate Hurst exponent using the Rescaled Range (R/S) method
    This is the proper method for financial time series
    """
    ts_np = np.array(ts).flatten()
    n = len(ts_np)
    
    if n < 50:
        return np.nan
    
    # Remove the mean to work with deviations
    ts_centered = ts_np - np.mean(ts_np)
    
    # Range of window sizes to test
    lags = range(10, min(n//4, 100))
    
    rs_values = []
    valid_lags_for_polyfit = []

    for lag in lags:
        # Number of non-overlapping windows
        n_windows = n // lag
        if n_windows < 2:
            continue
            
        rs_window_values = []
        
        for i in range(n_windows):
            # Extract window
            window = ts_centered[i*lag:(i+1)*lag]
            
            # Calculate cumulative sum (random walk)
            cumsum = np.cumsum(window)
            
            # Calculate range (max - min of cumulative sum)
            R = np.max(cumsum) - np.min(cumsum)
            
            # Calculate standard deviation of the window
            S = np.std(window, ddof=1)
            
            # Avoid division by zero
            if S > 1e-10:
                rs_window_values.append(R / S)
        
        if rs_window_values:
            # Average R/S across all windows for this lag
            rs_values.append(np.mean(rs_window_values))
            valid_lags_for_polyfit.append(lag)

    if len(rs_values) < 10:
        return np.nan

    rs_values = np.array(rs_values)
    
    # The relationship is: R/S ~ n^H
    # So log(R/S) = H * log(n) + constant
    log_lags = np.log(valid_lags_for_polyfit)
    log_rs = np.log(rs_values)
    
    # Linear regression to find H
    try:
        poly = np.polyfit(log_lags, log_rs, 1)
        hurst = poly[0]  # The slope is the Hurst exponent
        
        # Constrain to reasonable bounds
        hurst = max(0.0, min(1.0, hurst))
        
        return hurst
    except:
        return np.nan

def calculate_half_life(residuals):
    """
    Calculate half-life of mean reversion using AR(1) model
    """
    if len(residuals) < 2:
        return np.inf
    residuals_lagged = residuals.shift(1).dropna()
    residuals_current = residuals.iloc[1:]
    
    # AR(1) regression: y_t = a + b * y_{t-1} + error
    X = sm.add_constant(residuals_lagged)
    model = sm.OLS(residuals_current, X).fit()
    
    # Half-life = -ln(2) / ln(b) where b is the coefficient of the lagged term
    b = model.params.iloc[1]  # coefficient of lagged residual
    
    if b <= 0 or b >= 1:
        return np.inf  # No mean reversion or explosive
    
    half_life = -np.log(2) / np.log(b)
    return half_life

def test_cointegration_and_hurst(df1, df2, token1, token2, timeframe):
    """
    Runs the full analysis for a single pair and timeframe.
    """
    df1_filled = df1.ffill()
    df2_filled = df2.ffill()

    merged = pd.merge(
        df1_filled, df2_filled,
        left_index=True, right_index=True,
        how='inner',
        suffixes=(f'_{token1}', f'_{token2}')
    )
    merged.dropna(inplace=True)
    if len(merged) < 50:
        return None

    print(f"Checking {timeframe} {token1} and {token2}...", end=' ')

    merged[f'log_{token1}'] = np.log(merged[f'close_{token1}'])
    merged[f'log_{token2}'] = np.log(merged[f'close_{token2}'])

    X = sm.add_constant(merged[f'log_{token2}'])
    model = sm.OLS(merged[f'log_{token1}'], X).fit()
    merged['residual'] = model.resid

    adf_pvalue = adfuller(merged['residual'])[1]
    hurst = hurst_exponent(merged['residual'])
    half_life = calculate_half_life(merged['residual'])

    print(f"p = {adf_pvalue:.4f}, H = {hurst:.3f}, HL = {half_life:.1f}")
    
    return {
        'p_value': adf_pvalue,
        'hurst': hurst,
        'half_life': half_life,
    }

def load_data(tokens, timeframe, folder="pairs data"):
    data = {}
    for token in tokens:
        path = os.path.join(folder, f"{token}_{timeframe}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['open_time'], index_col='open_time')
            df = df[['close']].rename(columns={'close': f'close_{token}'})
            data[token] = df
    return data

def run_all_pairs(tokens, timeframe):
    print(f"\n--- Checking timeframe: {timeframe} ---")
    data = load_data(tokens, timeframe)
    results = []
    
    for token1, token2 in itertools.combinations(data.keys(), 2):
        result = test_cointegration_and_hurst(data[token1], data[token2], token1, token2, timeframe)
        if result:
            # FIX: Add identifying info to the result dictionary here
            result['pair_key'] = tuple(sorted((token1, token2)))
            result['timeframe'] = timeframe
            results.append(result)
    
    return results

def save_promising_pairs(all_results, filename="high_potential_pairs.txt"):
    """
    Filters and saves pairs that meet the strict criteria on at least one timeframe.
    The final output format now includes detailed stats for each promising timeframe.
    """
    pair_results = {}
    for result in all_results:
        # The KeyError happened here because 'pair_key' was not in result.
        # This is now fixed in the run_all_pairs function.
        pair_key = result['pair_key']
        if pair_key not in pair_results:
            pair_results[pair_key] = []
        pair_results[pair_key].append(result)

    kept_pairs_summary = {}

    half_life_limits = {
        '15m': 96, '1h': 72, '4h': 50, '1d': 21
    }

    for pair_key, results in pair_results.items():
        promising_timeframes_details = []
        for r in results:
            tf = r['timeframe']
            is_promising = (
                r.get('p_value') is not None and r['p_value'] < 0.05 and
                r.get('hurst') is not None and not np.isnan(r['hurst']) and r['hurst'] < 0.45 and
                r.get('half_life') is not None and not np.isinf(r['half_life']) and r['half_life'] < half_life_limits.get(tf, float('inf'))
            )
            
            if is_promising:
                promising_timeframes_details.append(r)
        
        if promising_timeframes_details:
            kept_pairs_summary[pair_key] = promising_timeframes_details

    if not kept_pairs_summary:
        print("\nNo pairs met the strict criteria for high-potential trading.")
        return

    with open(filename, 'w') as f:
        f.write("HIGH-POTENTIAL PAIRS FOR BACKTESTING\n")
        f.write("=====================================\n\n")
        
        sorted_pairs = sorted(kept_pairs_summary.items(), key=lambda item: item[0])
        
        for i, (pair_key, details) in enumerate(sorted_pairs):
            token1, token2 = pair_key
            f.write(f"{token1}/{token2}:\n")
            
            for detail in details:
                tf = detail['timeframe']
                p_val = detail['p_value']
                hurst = detail['hurst']
                hl = detail['half_life']
                
                tf_name_map = {'15m': '15 minute', '1h': '1 hour', '4h': '4 hour', '1d': '1 day'}
                tf_name = tf_name_map.get(tf, tf)
                
                f.write(f"  {tf_name}: p-value = {p_val:.4f}, Hurst = {hurst:.3f}, Half-life = {hl:.1f} periods\n")

            if i < len(sorted_pairs) - 1:
                 f.write("\n")

    print(f"\nSaved {len(kept_pairs_summary)} high-potential pairs to {filename}")
    print("These pairs and their specified timeframes meet the strict criteria for backtesting.")


if __name__ == "__main__":
    tokens = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT',
        'ARBUSDT', 'OPUSDT', 'MATICUSDT',
        'LDOUSDT', 'UNIUSDT', 'AAVEUSDT', 'CRVUSDT', 'COMPUSDT', 'DYDXUSDT', 'GMXUSDT', 'SUSHIUSDT',
        'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT', 'TRUMPUSDT',
        'FETUSDT', 'INJUSDT', 'LUNAUSDT', 'FTTUSDT'
    ]
    timeframes = ['15m', '1h', '4h', '1d']

    all_results = []
    for tf in timeframes:
        results = run_all_pairs(tokens, tf)
        all_results.extend(results)
    
    save_promising_pairs(all_results)