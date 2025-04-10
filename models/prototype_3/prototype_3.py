import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

class StockTradingEnv(gym.Env):
  def __init__(self, data, features, window_size=60, initial_balance=10000, volume=100, positions=None):
    super(StockTradingEnv, self).__init__()
    self.initial_balance = initial_balance
    self.equity = initial_balance
    self.balance = initial_balance
    self.max_position = 8
    self.data = data
    self.window_size = window_size
    self.current_step = window_size
    self.positions = positions
    self.volume = volume
    self.action_space = Discrete(3)
    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.window_size, features), dtype=np.float32)

  def step(self, action_signal):
    current_price = self.data['close'].iloc[self.current_step]
    # print(self.data.iloc[self.current_step])
    # print("Current step: ", self.current_step)
    # print("Current price: " + f"{current_price}")
    print(self.positions)
    reward = 0
    action = "nothing"

    positions = self.positions or []
    if action_signal == 0:
      action = "hold"
    elif action_signal == 1:
      if positions and positions[0]["type"] == "sell":
        action = "closeShort"
      elif len(positions) < self.max_position:
        action = "openLong"
      else:
        action = "hold"
    elif action_signal == 2:
      if positions and positions[0]["type"] == "buy":
        action = "closeLong"
      elif len(positions) < self.max_position:
        action = "openShort"
      else:
        action = "hold"

    if action == "hold":
      pass
    elif action == "openLong":
      self._open_position("buy", current_price)
    elif action == "openShort":
      self._open_position("sell", current_price)
    elif action == "closeLong":
      reward = self._close_positions("buy", current_price)
    elif action == "closeShort":      
      reward = self._close_positions("sell", current_price)

    reward += (self.balance - self.initial_balance) * 1e-4

    current_equity = self._calculate_equity(current_price)

    if current_equity < self.equity:
      reward -= (self.equity - current_equity) * 1e-3

    soft_loss_threshold = 0.01
    max_loss_threshold = 0.02

    for position in self.positions:
      entry_price = position["entry_price"]
      direction = position["type"]
      loss = (current_price - entry_price) if direction == "buy" else (entry_price - current_price)

      if loss >= 0:
        continue

      loss_ratio = abs(loss) / entry_price

      if loss_ratio > soft_loss_threshold:
        penalty_scale = 1e-3
        if loss_ratio > max_loss_threshold:
          penalty_scale *= 10
        reward -= loss_ratio * penalty_scale


    self.current_step += 1
    done = self.current_step >= len(self.data) - 1
    truncated = False

    state = self._get_state()
    return state, reward, done, truncated, {
      "balance": self.balance,
      "positions": len(self.positions),
      "reward": reward,
      "action": action
    }

  def reset(self, seed=None, **kwargs):
    if seed is not None:
      np.random.seed(seed)

    self.current_step = self.window_size
    self.balance = self.initial_balance
    self.equity = self.initial_balance

    if self.positions is None:
      self.positions = []

    return self._get_state(), {
      "positions": self.positions
    }

  def update_data(self, new_data):
    self.data = new_data
    # Rewind to last window if needed, but don't reset other state
    self.current_step = max(self.window_size, min(self.current_step, len(new_data) - 1))


  def _get_state(self):
    return self.data.iloc[self.current_step - self.window_size:self.current_step][
      ['EMA_12', 'EMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'ADX',
      'ADX_Positive', 'ADX_Negative', 'open', 'high', 'low', 'close', 'tick_volume']
    ].values

  def _open_position(self, position_type, current_price):
    position_cost = current_price * self.volume
    if self.balance >= position_cost:
      self.positions.append({
        "type": position_type,
        "entry_price": current_price,
        "entry_step": self.current_step
      })
      self.balance -= position_cost

  def _close_positions(self, position_type, current_price):
    total_profit = 0
    new_positions = []
    for position in self.positions:
      if position["type"] == position_type:
        entry_price = position["entry_price"]
        cost_basis = entry_price * self.volume
        profit = (entry_price - current_price) * self.volume if position_type == "buy" else (current_price - entry_price) * self.volume
        self.balance += cost_basis + profit
        total_profit += profit
      else:
        new_positions.append(position)
    self.positions = new_positions
    return total_profit

  def _calculate_equity(self, current_price):
    unrealized_pnl = sum(
      ((current_price - p["entry_price"]) * self.volume if p["type"] == "buy"
      else (p["entry_price"] - current_price) * self.volume)
      for p in self.positions
    )
    return self.balance + unrealized_pnl
  