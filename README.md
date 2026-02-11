# 🧠 Deep Reinforcement Learning for Cryptocurrency Portfolio Optimization
A deep reinforcement learning approach to dynamic cryptocurrency portfolio management using PPO and differential Sharpe ratio rewards.

## 📋 Overview
This project trains a Proximal Policy Optimization (PPO) agent to dynamically manage a 7-asset cryptocurrency portfolio, optimizing for risk-adjusted returns across multiple market regimes (2023-2025).

**Key Results:**

- Sharpe ratios: 0.90-3.15 across 4 out-of-sample test windows
- Outperformed baselines in 2 of 4 windows (volatile/sideways markets)
- Demonstrated adaptive learning but with high drawdowns (40-79%)


## 🎯 Key Features

- Algorithm: PPO with 2-layer MLP policy (Stable-Baselines3)
- Reward: Differential Sharpe ratio for online risk-adjusted learning
- Environment: Custom Gymnasium with 60-period lookback, log returns, and volatility features
- Assets: 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'ADAUSDT', 'NEOUSDT', 'XLMUSDT' (4-hour rebalancing)
- Validation: Walk-forward testing with strict temporal alignment (no data leakage)
- Baselines: Constrained MVO (10-50% position limits), BTC buy-and-hold


## 📊 Results

**Performance Summary (Out-of-Sample)**
|Window|Period      |Regime       |PPO Sharpe|MVO Sharpe  |BTC Sharpe  |PPO MaxDD  | Winner|
|------|------------|-------------|----------|------------|------------|-----------|-------|
|     1|Aug-Dec 2023|Strong Bull  | 1.22     |1.63        |  2.12      | -31.11%   | BTC   |
|     2|Jan-Jul 2024|Bull         | 0.90     |1.51        |  1.37      | -27.59%   | MVO   |
|     3|Aug-Dec 2024|Strong Bull  | 3.15     |2.01        |  2.27      | -31.18%   | PPO ✅|
|     4|Jan-Jul 2025|Sideways     | 1.84     |-0.33       |  0.69      | -33.05%   | PPO ✅|

Interpretation:

✅ PPO outperforms in sideways regime where MVO shows negative Sharpe.


## 🚀 Quick Start
**Installation**
```
git clone https://github.com/yeonjaej/drl-crypto-portfolio.git
cd drl-crypto-portfolio
pip install -r requirements.txt
```
**Run Analysis**
```
jupyter notebook notebooks/main_analysis.ipynb
```

## Technical details

### 🔧 Custom enviroment

**State Space** (600 features):
- Asset log returns: 7 assets × 60 timesteps (1 step = 4 hour)
- Volatility features: 3 indicators × 60 timesteps (σ_20, σ_60, market_vol)

**Action Space**: Continuous weights [-5, 5] → softmax normalized

**Reward**: Differential Sharpe ratio
```
d_sharpe = (b_t * Δa - 0.5 * a_t * Δb) / (b_t - a_t²)^1.5
where a_t = EMA(returns), b_t = EMA(returns²)
```

### Training 

- Vectorized: 4 parallel environments
- Total steps: 2M per window
- Batch size: 512
- Learning rate: 1e-4
- Clip range: 0.2
- GAE lambda: 0.95

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
```
## 🔮 Future Work

**High Priority:**

1. Transaction costs: Add 0.1-0.5% fees + slippage
2. Relax the fully-invested constraint (sum(w) = 1)
3. Position limits: Cap max allocation per asset (e.g., 40%)

Medium Priority:

4. Ensemble methods (PPO + MVO regime-switching)
5. Alternative rewards (Sortino, Calmar)
6. Additional features (sentiment, on-chain metrics)

Low Priority:

7. Alternative RL algorithms (SAC, TD3)
8. Different rebalancing frequencies (1H, daily, weekly)
9. Transfer learning from traditional assets

## 📚 References

> Srijan Sood, Kassiani Papasotiriou, Marius Vaiciulis, and Tucker Balch. (2023).
> *Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization*.
> In Proceedings of the ICAPS 2023 Workshop on AI in Finance (FinPlan).  
> [PDF](https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf)

## 📊 Key Takeaways

What Worked:

- Adaptive learning across market regimes
- Superior performance in volatile/sideways markets
- Differential Sharpe reward effective for online learning

What Didn't:

- Inconsistent performance across regimes
- No transaction cost modeling (overstates returns)

Lessons:

- Data leakage is subtle—requires careful validation
- RL excels in complex markets but isn't universally superior


⚠️ Disclaimer

- For educational purposes only. NOT investment advice.
- Backtest results do not guarantee future performance
- No transaction costs, slippage, or market impact modeled


👤 Author
Yeon-jae Jwa
📧 yeonjaejwa23@gmail.com  
💼 [LinkedIn](https://www.linkedin.com/in/yeon-jae-jwa)  
💻 [GitHub](https://github.com/yeonjaej)

📝 License
MIT License - see LICENSE file
