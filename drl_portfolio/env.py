# PortfolioEnv implementation here
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, data, vol_features, initial_cash=100000, lookback=60, max_steps=None):
        super(PortfolioEnv, self).__init__()
        self.data = data  # shape: (T_total, n_assets)
        self.vol_features = vol_features  # shape: (T_total, 3) -> [vol20, vol20/vol60, VIX]
        self.lookback = lookback
        self.n_assets = self.data.shape[1]
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=((self.n_assets + 4) * lookback,), dtype=np.float32)
        self.max_steps = max_steps or (len(self.data) - self.lookback - 1)

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
        self.portfolio_value = 1.0
        self.a_t = 0.0  # EMA of returns
        self.b_t = 0.0  # EMA of squared returns
        self.eta = 1 / (365*6)
        self.steps = 0
        self.episode_reward = 0.0

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

        # Update differential Sharpe components
        delta_a = portfolio_return - self.a_t
        delta_b = portfolio_return**2 - self.b_t
        self.a_t += self.eta * delta_a
        self.b_t += self.eta * delta_b


        if self.b_t - self.a_t**2 > 1e-6:
            d_sharpe = (self.b_t * delta_a - 0.5 * self.a_t * delta_b) / (self.b_t - self.a_t**2)**1.5
        else:
            d_sharpe = 0.0

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

        # Broadcast cash + vol features across lookback
        cash_row = np.full((1, self.lookback), self.cash, dtype=np.float32)
        #cash_row = np.tile(np.array([self.cash] + list(vol_feats)).reshape(-1, 1), (1, self.lookback))  # shape: (4, lookback)

        # Combine: (n_assets + 1 row, lookback columns)
        state = np.vstack([returns, vol_history, cash_row])  # final shape: (n_assets + 4, lookback)

        return state.flatten().astype(np.float32)

