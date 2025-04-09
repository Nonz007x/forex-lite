import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.spaces import Discrete, Box
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class StockTradingEnv(gym.Env):
  def __init__(self, data, window_size=60, initial_balance=10000, volume=100, positions=[]):
    super(StockTradingEnv, self).__init__()
    self.initial_balance = initial_balance
    self.equity = initial_balance
    self.balance = initial_balance
    self.max_drawdown_pct = 0.1
    self.max_position = 8
    self.data = data
    self.window_size = window_size
    self.current_step = window_size
    self.positions = positions
    self.volume = volume
    self.action_space = Discrete(3)
    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.window_size, 14), dtype=np.float32)

  def step(self, action_signal):
    current_price = self.data['close'].iloc[self.current_step]
    reward = 0
    action = ""

    current_drawdown = (self.initial_balance - self.balance) / self.initial_balance
    if current_drawdown > self.max_drawdown_pct:
      state = self._get_state
      return state, reward, True, False, {
      "balance": self.balance,
      "positions": len(self.positions),
      "reward": reward
    }

    if action == 0:
      action = "hold"
    elif action_signal == 1:
      if self.positions and self.positions[0]["type"] == "sell":
        action = "closeShort"
      elif len(self.positions) < self.max_position:
        action = "openLong"
      else:
        action = "hold"
    elif action_signal == 2:
      if self.positions and self.positions[0]["type"] == "buy":  
        action = "closeLong"
      elif len(self.positions ) < self.max_position:
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
      reward -= (self.equity - current_equity) * 5e-4

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
      "reward": reward
    }

  def reset(self, seed=None, **kwargs):
    if seed is not None:
      np.random.seed(seed)  # Optional: Set the seed for randomness

    self.current_step = self.window_size
    self.balance = 10000
    self.positions = []
    return self.data.iloc[self.current_step - self.window_size:self.current_step].values, {}

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
  
  
def load_data_from_csv(csv_path):
  df = pd.read_csv(csv_path, header=None)
  column_names = ["date", "time", "open", "high", "low", "close", "tick_volume", "volume", "spread"]
  df.columns = column_names[:df.shape[1]]
  return df

def create_indicator(df): 
    if 'date' in df.columns:
      df = df.drop(columns=['date'])
    if 'time' in df.columns:
      df = df.drop(columns=['time'])

    df = df.apply(pd.to_numeric, errors='coerce')

    df['EMA_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
    df['EMA_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['MACD'] = MACD(df['close']).macd()
    df['RSI'] = RSIIndicator(df['close']).rsi()

    bb = BollingerBands(df['close'], window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_indicator.adx()
    df['ADX_Positive'] = adx_indicator.adx_pos()
    df['ADX_Negative'] = adx_indicator.adx_neg()

    df.dropna(inplace=True)

    return df[['EMA_12', 'EMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'ADX', 'ADX_Positive', 'ADX_Negative',  'open', 'high', 'low', 'close', 'tick_volume']]

def make_env():
  return StockTradingEnv(df[:149002])
  
if __name__ == '__main__':

  parent_dir = Path(__file__).parent.parent.parent
  csv_path = parent_dir / "train-data" / "eurusd.csv"
  data = load_data_from_csv(csv_path)

  df = create_indicator(data)

  # env = DummyVecEnv([lambda: StockTradingEnv(df[:149002])])
  env = SubprocVecEnv([make_env for _ in range(12)])

  model = PPO('MlpPolicy',
            env,
            verbose=1,
            learning_rate=8e-5,
            gamma=0.98,
            gae_lambda=0.945,
            ent_coef=0.01,  
            n_epochs=10,
            vf_coef=0.5,
            clip_range=0.2,
            batch_size=1024,
            n_steps=2048,
            policy_kwargs=dict(net_arch=[256, 128, 64])
           )

  model.learn(total_timesteps=122880)
  print("finished training", datetime.now())
  script_dir = Path(__file__).parent
  file_path = script_dir / "model"
  model.save(str(file_path))
