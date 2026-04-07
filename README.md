# Deep Reinforcement Learning for Cryptocurrency Portfolio Optimization

This repository explores deep reinforcement learning for multi-asset crypto portfolio allocation. It trains a Proximal Policy Optimization (PPO) agent in a custom Gymnasium environment and evaluates it against classical baselines such as constrained mean-variance optimization (MVO) and BTC buy-and-hold.

The project is notebook-driven: [notebooks/main_analysis.ipynb](/Users/jwayeonjae/Downloads/drl-crypto-portfolio/notebooks/main_analysis.ipynb) is the main entry point, while the `drl_portfolio` package contains the reusable environment, feature engineering, and plotting utilities.

## What The Project Does

At a high level, the pipeline is:

1. Load 4-hour historical crypto prices from CSV or download them from Binance.
2. Generate rolling market features such as log returns, volatility, trend distance, and cross-asset correlation.
3. Feed those features into a custom portfolio environment with a cash position and transaction costs.
4. Train PPO to output portfolio weights that maximize a differential Sharpe-style reward.
5. Evaluate each trained model on unseen windows and compare against MVO and BTC buy-and-hold.

## Current Setup

The current codebase reflects a more recent experiment setup than some older project notes:

- Frequency: 4-hour bars
- Data range used in the notebook: 2023-01-01 through 2026-04-01
- Asset universe in `crypto_data_v4.csv`: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `NEARUSDT`, `AVAXUSDT`, `OPUSDT`, `LDOUSDT`, `FETUSDT`, `LINKUSDT`, `AAVEUSDT`, `UNIUSDT`, `DOGEUSDT`
- Action space: one cash weight plus one weight per crypto asset
- Transaction costs in the environment: 0.10% trading fee + 0.05% slippage
- Baselines: constrained MVO and BTC buy-and-hold

## Repository Structure

```text
drl-crypto-portfolio/
├── data/
│   ├── crypto_data.csv
│   ├── crypto_data_v2.csv
│   ├── crypto_data_v3.csv
│   └── crypto_data_v4.csv
├── drl_portfolio/
│   ├── __init__.py
│   ├── env.py
│   ├── features.py
│   └── utils.py
├── notebooks/
│   ├── images/
│   └── main_analysis.ipynb
├── results/
│   ├── best_model_window_*.zip
│   ├── vec_norm_window_*.pkl
│   └── ppo_logs/
├── requirements.txt
└── README.md
```

## Core Modules

### `drl_portfolio/env.py`

Defines `PortfolioEnv`, the custom Gymnasium environment used for training and evaluation.

Key behaviors:

- Adds an explicit cash position at action index `0`
- Converts raw PPO actions into normalized portfolio weights using softmax
- Applies turnover-based transaction costs
- Tracks drifting portfolio weights after market movement
- Uses a clipped differential Sharpe reward for online RL training

### `drl_portfolio/features.py`

Builds the model inputs from price data.

Current engineered features include:

- asset log returns
- short and long rolling volatility
- volatility ratio
- cross-asset volatility proxy
- trend distance from a long moving average
- rolling mean cross-asset correlation

The function returns aligned prices and features so the environment can step through them without leakage from missing lookback periods.

### `drl_portfolio/utils.py`

Contains helper functions for:

- generating rolling train/test windows
- plotting PPO portfolio weights over time
- summarizing weight statistics

## Training And Evaluation Workflow

The notebook performs the full experiment loop:

1. Prepare price history and clean missing columns.
2. Generate features for the full dataset.
3. Create expanding train, validation, and test windows.
4. Train one PPO model per window using vectorized environments.
5. Save the best checkpoint and normalization statistics for each window.
6. Run out-of-sample evaluation on the corresponding test window.
7. Compare PPO against:
   - constrained MVO using PyPortfolioOpt
   - BTC buy-and-hold

Important PPO settings used in the current notebook:

- Policy: `MlpPolicy`
- Vectorized training envs: 4
- Timesteps per window: 8,000,000
- `n_steps`: 2048
- `batch_size`: 512
- `n_epochs`: 5
- `gae_lambda`: 0.95
- `clip_range`: 0.2
- learning rate schedule starting at `5e-5`

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/yeonjaej/drl-crypto-portfolio.git
cd drl-crypto-portfolio
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

The notebook can either:

- use existing CSV data already stored in `data/`
- or download fresh Binance history and save a new dataset

## Data Files

There are multiple CSV snapshots in `data/`, which reflect different iterations of the experiment. The latest dataset in this repository appears to be `data/crypto_data_v4.csv`, while the notebook currently defaults to loading `data/crypto_data_v3.csv` when the download flag is disabled.

If you switch datasets, make sure the asset universe and any saved models in `results/` still match the environment dimensions.

## Saved Artifacts

The `results/` directory contains prior experiment outputs, including:

- saved PPO checkpoints for multiple windows
- `VecNormalize` statistics used during evaluation
- TensorBoard training logs
- older model/log folders from earlier experiment versions

These artifacts are useful for reproducing evaluation without retraining from scratch.

## Caveats

This is a research/backtesting project, not a live trading system.

Current limitations include:

- notebook-centric workflow rather than a packaged training CLI
- no automated test suite for the RL pipeline
- experiment versions and documentation have evolved over time
- backtest assumptions may still differ from real execution conditions

## License

MIT License. See [LICENSE](/Users/jwayeonjae/Downloads/drl-crypto-portfolio/LICENSE).
