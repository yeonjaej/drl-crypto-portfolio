# 🧠 Deep Reinforcement Learning for Cryptocurrency Portfolio Optimization
A deep reinforcement learning approach to dynamic cryptocurrency portfolio management using PPO and differential Sharpe ratio rewards.

## 📋 Overview
This project trains a Proximal Policy Optimization (PPO) agent to dynamically manage a 5-asset cryptocurrency portfolio, optimizing for risk-adjusted returns across multiple market regimes (2023-2025).
Key Results:

Sharpe ratios: 0.90-3.15 across 4 out-of-sample test windows
Outperformed baselines in 2 of 4 windows (volatile/sideways markets)
Demonstrated adaptive learning but with high drawdowns (40-79%)


## 🎯 Key Features

Algorithm: PPO with 2-layer MLP policy (Stable-Baselines3)
Reward: Differential Sharpe ratio for online risk-adjusted learning
Environment: Custom Gymnasium with 60-period lookback, log returns, and volatility features
Assets: BTC, ETH, BNB, SOL, ADA (4-hour rebalancing)
Validation: Walk-forward testing with strict temporal alignment (no data leakage)
Baselines: Constrained MVO (10-50% position limits), BTC buy-and-hold, equal-weight


## 📊 Results
Performance Summary (Out-of-Sample)
|Window|Period      |Regime       |PPO Sharpe|MVO Sharpe  |BTC Sharpe  |PPO MaxDD  | Winner|
|------|------------|-------------|----------|------------|------------|-----------|-------|
|     1|Aug-Dec 2023|Bear→Recovery| 1.22     |1.63        |  2.12      | -43.50%   | BTC   |
|     2|Jan-Jul 2024|Bull         | 0.90     |1.51        |  1.37      | -42.25%   | MVO   |
|     3|Aug-Dec 2024|Strong Bull  | 3.15     |2.01        |  2.27      | -78.86%   | PPO ✅|
|     4|Jan-Jul 2025|Sideways     | 1.84     |-0.33       |  0.69      | -59.22%   | PPO ✅|

Interpretation:

✅ PPO excels in trending (Window 3) and choppy (Window 4) markets
⚠️ Underperforms during smooth recoveries (Windows 1-2)
🚨 Excessive drawdowns (-79% max) require risk constraints for production


🚀 Quick Start
Installation
bashgit clone https://github.com/yeonjaej/drl-crypto-portfolio.git
cd drl-crypto-portfolio
pip install -r requirements.txt
Run Analysis
bashjupyter notebook notebooks/main_analysis.ipynb
Train Custom Agent
pythonfrom drl_portfolio.environment import PortfolioEnv
from stable_baselines3 import PPO

env = PortfolioEnv(data=prices, vol_features=vols, max_steps=1024)
model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
model.learn(total_timesteps=2_000_000)
```

---

## 🔧 Technical Details

### Environment

**State Space** (540 features):
- Asset log returns: 5 assets × 60 timesteps
- Volatility features: 3 indicators × 60 timesteps (σ₂₀, σ₆₀, ratio)
- Cash position: 1 × 60 timesteps

**Action Space**: Continuous weights [-5, 5] → softmax normalized

**Reward**: Differential Sharpe ratio
```
d_sharpe = (b_t * Δa - 0.5 * a_t * Δb) / (b_t - a_t²)^1.5
where a_t = EMA(returns), b_t = EMA(returns²)
Training

Vectorized: 4 parallel environments
Total steps: 2M per window
Batch size: 512
Learning rate: 1e-4
Clip range: 0.2
GAE lambda: 0.95

Data Leakage Prevention
python# CORRECT: Observation at time t uses only data up to t-1
obs = returns[t - lookback : t]  # [t-60, ..., t-1]

# WRONG (causes leakage):
obs = returns[t - lookback : t+1]  # Includes t (future!)
```

Validated via: random policy achieves ~0 Sharpe, proving no trivial exploitation.

---

## 📁 Repository Structure
```
drl-crypto-portfolio/
├── drl_portfolio/          # Core modules
│   ├── environment.py      # Custom Gymnasium env
│   └── features.py         # Feature engineering
├── notebooks/              # Analysis notebooks
│   └── main_analysis.ipynb # End-to-end pipeline
├── results/                
│   ├── models/             # Trained PPO agents
│   └── ppo_logs/           # TensorBoard logs
└── requirements.txt

🔮 Future Work
High Priority (Production Readiness):

Transaction costs: Add 0.1-0.5% fees + slippage
Drawdown constraints: Implement max drawdown limits or penalize in reward
Position limits: Cap max allocation per asset (e.g., 40%)

Medium Priority:
4. Ensemble methods (PPO + MVO regime-switching)
5. Alternative rewards (Sortino, Calmar)
6. Additional features (sentiment, on-chain metrics)
Low Priority:
7. Alternative RL algorithms (SAC, TD3)
8. Different rebalancing frequencies (1H, daily, weekly)
9. Transfer learning from traditional assets

📚 References
Primary:

Sood et al. (2023). Deep Reinforcement Learning for Optimal Portfolio Allocation.
ICAPS Workshop on AI in Finance. PDF

Methodology:

Differential Sharpe: Moody & Saffell (2001)
PPO Algorithm: Schulman et al. (2017)
MVO: Markowitz (1952)


📊 Key Takeaways
What Worked:

Adaptive learning across market regimes
Superior performance in volatile/sideways markets
Differential Sharpe reward effective for online learning

What Didn't:

Excessive drawdowns (max -79%)
Inconsistent performance across regimes
No transaction cost modeling (overstates returns)

Lessons:

Data leakage is subtle—requires careful validation
Constrained baselines essential for fair comparison
Sharpe alone insufficient—must consider drawdown, turnover
RL excels in complex markets but isn't universally superior


⚠️ Disclaimer
For educational purposes only. NOT investment advice.

Backtest results do not guarantee future performance
No transaction costs, slippage, or market impact modeled
High drawdowns make this unsuitable for live trading without modifications


👤 Author
Yeon-jae Jwa
📧 yeonjaejwa23@gmail.com | 🔗 LinkedIn | 💻 GitHub

📝 License
MIT License - see LICENSE file

⭐ Star this repo if you found it useful!Claude is AI and can make mistakes. Please double-check responses.

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

- Add trading fees to the environment
- 
- Penalize over-concentration and volatility exposure
- Explore alternative reward functions
- Compare across rebalancing frequencies
- Explore non-fully invested stratege