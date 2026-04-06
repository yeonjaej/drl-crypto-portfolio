# PortfolioEnv implementation here
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    portfolio env
    - Cash position (index 0)
    - Transaction costs (fees + slippage)
    - Position tracking
    """
    TRADING_FEE = 0.001      # 0.1% taker fee
    SLIPPAGE = 0.0005        # 0.05% estimated slippage
    TOTAL_COST = TRADING_FEE + SLIPPAGE  # 0.15% per side
    N_VOL = 3
    N_CASH = 1

    def __init__(self, data, vol_features, regime_features, initial_cash=1., lookback=60, max_steps=None):
        super(PortfolioEnv, self).__init__()
        self.data = data  # shape: (T_total, n_assets)
        self.vol_features = vol_features  # shape: (T_total, 3) -> [vol20, vol20/vol60, VIX]
        self.regime_features = regime_features
        self.lookback = lookback
        self.n_assets = self.data.shape[1]
        self.n_actions = self.n_assets + self.N_CASH # +1 for cash position

        # allowed action space for fully invested portfolio
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n_actions,), dtype=np.float32)
        
        # lookback for log returns / vol
        features_per_step = self.n_assets + self.N_VOL + self.n_assets + self.N_CASH
        obs_size = (features_per_step)* lookback + self.n_actions

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_size,), dtype=np.float32) 
        
        self.max_steps = max_steps or (len(self.data) - self.lookback - 1)
        self.initial_cash = initial_cash

        # Internal state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # randomize start step
        latest_possible_start = len(self.data) - self.max_steps -1
        if latest_possible_start > self.lookback:
            self.t = np.random.randint(self.lookback, latest_possible_start)
        else:
            self.t = self.lookback
        
        self.portfolio_value = self.initial_cash
        self.eta = 1 / (365*6)
        #self.eta = 2/(self.lookback+1)
        self.steps = 0
        self.episode_reward = 0.0
        self.cumulative_cost = 0.0

        # start fully in cash
        self.weights = np.zeros(self.n_actions)
        self.weights[0] = 1.0

        # --- WARM-UP INITIALIZATION ---
        # Calculate the initial mean (a_t) and second moment (b_t) 
        # using the lookback window so the agent starts with valid stats.
        
        # Get data from (t - lookback) to t
        past_data = self.data[self.t - self.lookback : self.t]
        
        # Calculate simple returns for this period to estimate stats
        # (Assuming simple returns for Sharpe, consistent with your step function)
        past_returns = (past_data[1:] - past_data[:-1]) / past_data[:-1]
        
        # Assume an equal weight portfolio for initialization statistics
        # (Or you can start with all zeros, but stats need to reflect market magnitude)
        init_weights = np.ones(self.n_assets) / self.n_assets
        portfolio_past_ret = np.dot(past_returns, init_weights)

        self.a_t = 0.0 #np.mean(portfolio_past_ret) # EMA (return)
        self.b_t = 0.0 # np.mean(portfolio_past_ret**2) # EMA *(return**2)
        # -----------------------------------


        return self._get_observation(), {}

    def step(self, action):
        if self.t + 1 >= len(self.data):
            return self._get_observation(), 0.0, True, False, {}
        # Normalize weights using softmax
        target_weights = np.exp(action - np.max(action))
        target_weights /= np.sum(target_weights)

        # turnover= sum of absolute weight changes / 2
        # (dividing by 2 because every sell has a correspoinding buy)
        weight_diff = np.abs(target_weights - self.weights)
        turnover = np.sum(weight_diff) / 2.0 
        transaction_cost = turnover * self.TOTAL_COST
        self.cumulative_cost += transaction_cost

        # Compute portfolio return
        prev_prices = self.data[self.t]
        curr_prices = self.data[self.t + 1]
        asset_returns = (curr_prices - prev_prices) / prev_prices

        # cash return = 0,
        crypto_weights = target_weights[1:]
        portfolio_return = np.dot(crypto_weights, asset_returns)

        portfolio_return -= transaction_cost

        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)

        #   Drift weights (market movement changes weights) ────
        #   After the market moves, actual weights shift.
        #   Cash weight stays, crypto weights scale by (1 + r_i)
        drifted = target_weights.copy()
        drifted[1:] = target_weights[1:] * (1 + asset_returns)
        drifted /= np.sum(drifted)
        self.weights = drifted
        

        # ----------------------------------------------------------------------
        # DIFFERENTIAL SHARPE RATIO (Moody & Saffell, 2001)
        # ----------------------------------------------------------------------
        # The goal is to maximize the Sharpe Ratio (Mean / StdDev).
        # Since standard Sharpe is calculated over the whole episode, we need
        # a "per-step" proxy to train the RL agent online.
        #
        # D_t represents the contribution of the CURRENT step's return to the 
        # long-term Sharpe Ratio.
        #
        # Variables:
        #   A_t: Exponential Moving Average (EMA) of returns (estimated Mean)
        #   B_t: EMA of squared returns (estimated Second Moment)
        #   delta_a: Deviation of current return from the mean (R_t - A_{t-1})
        #   delta_b: Deviation of current sq. return from 2nd moment (R_t^2 - B_{t-1})
        # ----------------------------------------------------------------------

        

        # 1. Calculate how the current return (portfolio_return) shifts our moving averages
        #    Note: We use the stats from the PREVIOUS step (self.a_t, self.b_t)
        delta_a = portfolio_return - self.a_t
        delta_b = portfolio_return**2 - self.b_t
        
        # prev_variance = max(self.b_t - self.a_t**2, 1e-4)
        prev_variance = max(self.b_t - self.a_t**2, 1e-3) # higher variance floor for full fee env
        
        d_sharpe = (self.b_t * delta_a - 0.5 * self.a_t * delta_b) / (prev_variance**1.5)

        self.a_t += self.eta * delta_a
        self.b_t += self.eta * delta_b
        
        cash_penalty_rate = 0.05 
        cash_burn = self.weights[0] * cash_penalty_rate * self.eta

        # Subtract the cash burn from the Sharpe reward
        final_reward = d_sharpe # - cash_burn # cash_burn for full fee env
        
        # Clip for stable PPO training
        # reward = np.clip(final_reward, -2.0, 2.0)
        reward = np.clip(final_reward, -1.0, 1.0) # tighter clip for full fee env

        self.episode_reward += reward
        
        self.t += 1
        self.steps += 1
        
        terminated = self.t >= len(self.data) - 1  
        truncated = self.steps >= self.max_steps
        info = {}

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,      # total reward
                "l": self.steps,
                "final_value": self.portfolio_value,                                # episode length
                "cumulative_cost": self.cumulative_cost,
            }
        # in Gymnasium, step() must return exactly below
        # 1st return val; observation:  agent "Sees it."
        # 2nd return val; reward:       agent "Maximizes it."
        # 3rd, 4th return val; terminated/truncated:   agent "Checks it".
        # 5th return val; info:         for book-keeping.
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        # Get log returns: shape (lookback, n_assets)
        returns = np.log(self.data[self.t - self.lookback + 1:self.t + 1] /
                        self.data[self.t - self.lookback:self.t])  # shape: (lookback, n_assets)
        returns = np.nan_to_num(returns)

        # Transpose to shape (n_assets, lookback)
        returns = returns.T  # shape (n_assets, lookback)

        # Get volatility features at current timestep
        vol_history = self.vol_features[self.t - self.lookback + 1: self.t +1]
        # vol_feats = self.vol_features[self.t]  # shape: (3,)
        vol_history = vol_history.T # shape(lookback, 3)

        regime_history = self.regime_features[self.t - self.lookback + 1: self.t + 1].T

        market_features = np.vstack([returns, vol_history, regime_history]).flatten()
        portfolio_state = self.weights.copy()
        obs = np.concatenate([market_features, portfolio_state])
        return obs.astype(np.float32)
        #state = np.vstack([returns, vol_history])

        #return state.flatten().astype(np.float32)

