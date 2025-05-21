# 🧠 DRL Crypto Portfolio Optimization

This repository demonstrates the use of Deep Reinforcement Learning (PPO) to optimize a cryptocurrency portfolio over time using a custom OpenAI Gymnasium environment and a differential Sharpe ratio-based reward.

## 📈 Project Overview

- **Environment**: `PortfolioEnv` simulates portfolio value with price and volatility features
- **RL Agent**: PPO (from Stable-Baselines3), trained in vectorized environments
- **Reward**: Differential Sharpe ratio
- **Baselines**: Mean-Variance Optimization (MVO), BTC Buy-and-Hold
- **Data**: Crypto asset prices and volatilities at 4-hour frequency

## 📊 Results Summary

PPO outperformed MVO and BTC-hold in 2 of 3 market windows in terms of Sharpe ratio. The agent showed adaptability across different regimes, especially in trending or high-volatility markets.

## 🏗️ Repository Structure

```
drl-crypto-portfolio/
│
├── drl_portfolio/          # Core Python modules (env, train, eval)
├── notebooks/              # Main analysis notebook
├── results/                # TensorBoard logs, trained models, plots
├── data/                   # Preprocessed price and feature files
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore
```

## 🚀 Quickstart

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main analysis:
    ```bash
    jupyter notebook notebooks/main_analysis.ipynb
    ```

3. Train or load PPO agents, evaluate them on test windows, and visualize results.

## 🧪 Evaluation Metrics

- **Sharpe Ratio**
- **Max Drawdown**
- **Cumulative Return**

## 📚 Reference

> Srijan Sood, Kassiani Papasotiriou, Marius Vaiciulis, and Tucker Balch. (2023).
> *Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization*.
> In Proceedings of the ICAPS 2023 Workshop on AI in Finance (FinPlan).  
> [PDF](https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf)

## 🔮 Future Directions

- Add trading fees & slippage to the environment
- Penalize over-concentration and volatility exposure
- Explore alternative reward functions
- Compare across rebalancing frequencies

MIT License | Built with ❤️  by @yeonjaej@github.com