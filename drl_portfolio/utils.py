# Utility functions like Sharpe, date slicing
import pandas as pd
from datetime import datetime, timedelta
#import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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

        windows.append((train_start, train_end, test_start, test_end))

        #    train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
        #    test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")
        #))

        # Slide by test days 
        current_start += pd.Timedelta(days=test_days)

    return windows

def plot_ppo_weights_over_time(results, window_idx, asset_names=None):
    """
    Plot how PPO adjusts portfolio weights over time.
    
    Args:
        results: List of result dictionaries from PPO evaluation
        window_idx: Which window to plot (0, 1, 2, ...)
        asset_names: List of asset names (e.g., ['BTCUSDT', 'ETHUSDT', ...])
    """
    res = results[window_idx]
    weights = res["weights"]  # shape: (T, n_assets)
    dates = res["dates"]
    
    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(weights.shape[1])]
    
    n_assets = weights.shape[1]
    
    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_assets))
    
    # Plot stacked area
    ax.stackplot(dates, weights.T, labels=asset_names, colors=colors, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.set_title(f'Window {res["window"]}: PPO Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def plot_weight_statistics(results, window_idx, asset_names=None):
    """
    Show summary statistics of weights: mean, std, min, max.
    """
    res = results[window_idx]
    weights = res["weights"]
    
    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(weights.shape[1])]
    
    # Calculate statistics
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)
    min_weights = weights.min(axis=0)
    max_weights = weights.max(axis=0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(asset_names))
    width = 0.6
    
    # Plot bars with error bars
    bars = ax.bar(x, mean_weights, width, yerr=std_weights, 
                   capsize=5, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(asset_names))))
    
    # Add min/max indicators
    for i, (mn, mx) in enumerate(zip(min_weights, max_weights)):
        ax.plot([i, i], [mn, mx], 'k-', alpha=0.3, linewidth=2)
        ax.plot(i, mn, 'kv', markersize=6, alpha=0.5)
        ax.plot(i, mx, 'k^', markersize=6, alpha=0.5)
    
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title(f'Window {res["window"]}: PPO Weight Statistics\n(Mean ± Std, with Min/Max range)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=bars[0].get_facecolor(), alpha=0.8, label='Mean'),
        Line2D([0], [0], color='k', linewidth=2, alpha=0.3, label='Min-Max Range'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8, alpha=0.5, label='Max', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='k', markersize=8, alpha=0.5, label='Min', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics table
    print(f"\n📊 Weight Statistics for Window {res['window']}:")
    print(f"{'Asset':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    for i, name in enumerate(asset_names):
        print(f"{name:<15} {mean_weights[i]:>7.2%} {std_weights[i]:>7.2%} {min_weights[i]:>7.2%} {max_weights[i]:>7.2%}")
