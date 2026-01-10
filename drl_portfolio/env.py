# PortfolioEnv implementation here
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, data, vol_features, initial_cash=1, lookback=60, max_steps=None):
        super(PortfolioEnv, self).__init__()
        self.data = data  # shape: (T_total, n_assets)
        self.vol_features = vol_features  # shape: (T_total, 3) -> [vol20, vol20/vol60, VIX]
        self.lookback = lookback
        self.n_assets = self.data.shape[1]

        # allowed action space for fully invested portfolio
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n_assets,), dtype=np.float32)
        
        # lookback for log returns / vol
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=((self.n_assets + 3) * lookback,), dtype=np.float32) 
        
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
        
        self.cash = 1.0  # normalize portfolio to 1.0
        self.weights = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_cash
        self.eta = 1 / (365*6)
        self.steps = 0
        self.episode_reward = 0.0

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

        self.a_t = np.mean(portfolio_past_ret) # EMA (return)
        self.b_t = np.mean(portfolio_past_ret**2) # EMA *(return**2)
        # -----------------------------------


        return self._get_observation(), {}

    def step(self, action):
        if self.t + 1 >= len(self.data):
            return self._get_observation(), 0.0, True, False, {}
        # Normalize weights using softmax
        weights = np.exp(action)
        weights /= np.sum(weights)

        # Compute portfolio return
        prev_prices = self.data[self.t]
        curr_prices = self.data[self.t + 1]
        returns = (curr_prices - prev_prices) / prev_prices
        portfolio_return = np.dot(weights, returns)

        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)

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

        #2. Calculate Variance: Var = E[x^2] - (E[x])^2
        prev_variance = self.b_t - self.a_t**2

        # 3. Calculate Reward
        #    Formula: (B * delta_A - 0.5 * A * delta_B) / (Variance ^ 1.5)
        #    Term 1 (B * delta_a): Rewards returns that drive the average up.
        #    Term 2 (-0.5 * A * delta_b): Penalizes returns that increase variance (risk).
        if prev_variance > 1e-6:
            d_sharpe = (self.b_t * delta_a - 0.5 * self.a_t * delta_b) / (prev_variance**1.5)
        else:
            d_sharpe = 0.0

        # 4. Update the Moving Averages for the NEXT step
        #    eta is the "learning rate" of the moving average (forgetting factor)
        self.a_t += self.eta * delta_a
        self.b_t += self.eta * delta_b
        
        reward = d_sharpe

        self.episode_reward += reward
        self.weights = weights
        self.t += 1
        self.steps +=1
        
        terminated = self.t >= len(self.data) - 1 or self.steps >= self.max_steps
        truncated = False
        info = {}

        if terminated:
            info["episode"] = {
                "r": self.episode_reward,      # total reward
                "l": self.steps,
                "final_value": self.portfolio_value                                # episode length
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

        state = np.vstack([returns, vol_history])

        return state.flatten().astype(np.float32)

