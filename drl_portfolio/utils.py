# Utility functions like Sharpe, date slicing
import pandas as pd
from datetime import datetime, timedelta
#import time

def generate_time_windows(price_df, train_years=3.5, test_years=0.5):
    """
    Generate sliding windows for training and testing.
    Supports fractional year values (e.g., 0.5 = 6 months).
    
    Args:
        price_df (pd.DataFrame): Must have a DateTimeIndex.
        train_years (float): Length of training window in years.
        test_years (float): Length of test window in years.
    
    Returns:
        List of (train_start, train_end, test_start, test_end) tuples.
    """
    df = price_df.copy()
    df = df[~df.index.duplicated()].sort_index()
    
    min_date = df.index.min()
    max_date = df.index.max()

    # Convert fractional years into days (approx)
    def years_to_days(years):
        return int(years * 365.25)  # includes leap year adjustment

    train_days = years_to_days(train_years)
    test_days = years_to_days(test_years)
    total_days = train_days + test_days

    windows = []
    current_start = min_date

    while current_start + pd.Timedelta(days=total_days) <= max_date:
        train_start = current_start
        train_end = train_start + pd.Timedelta(days=train_days - 1)

        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        windows.append((
            train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")
        ))

        # Slide by test days 
        current_start += pd.Timedelta(days=test_days)

    return windows
