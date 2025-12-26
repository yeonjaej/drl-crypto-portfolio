import numpy as np
import pandas as pd
# Feature generation functions here
def generate_features(price_df, lookbacks=(20, 60)):
    """
    Generate log returns and volatility features for crypto data.

    Args:
        price_df (pd.DataFrame): OHLCV close prices, DateTimeIndex, with given frequency (e.g. 4H).
        lookbacks (tuple): Lookback periods for volatility (e.g. (20, 60)).

    Returns:
        log_returns: DataFrame of log returns
        vol_feats: np.array of shape (len(log_returns), 3) -> [vol20, vol_ratio, vix_proxy]
        aligned_prices: np.array of prices aligned with vol_feats
    """
    # 1. Log returns
    log_returns = np.log(price_df / price_df.shift(1)).dropna()

    # 2. Rolling volatilities
    vol_short = log_returns.rolling(lookbacks[0]).std().mean(axis=1)
    vol_long = log_returns.rolling(lookbacks[1]).std().mean(axis=1)

    # 3. Volatility ratio and VIX proxy
    vol_ratio = vol_short / vol_long
    vix_proxy = log_returns.std(axis=1) * 100  # cross-asset std dev

    # 4. Stack features into a matrix
    vol_feats_df = pd.DataFrame(np.stack([vol_short, vol_ratio, vix_proxy], axis=1),
                                 index=log_returns.index)
    
    # 5. Drop the rows that don't have enough history.
    vol_feats_df = vol_feats_df.dropna()

    # 6. Align prices and features
    valid_index = vol_feats_df.index
    log_returns = log_returns.loc[valid_index]
    aligned_prices = price_df.loc[valid_index].values
    vol_feats = vol_feats_df.values

    return log_returns, vol_feats, aligned_prices, valid_index