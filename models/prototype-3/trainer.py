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
  def __init__(self, data, window_size=60, initial_balance=10000, volume=100):
    super(StockTradingEnv, self).__init__()
    self.initial_balance = initial_balance
    self.max_drawdown_pct = 0.1
    self.data = data
    self.window_size = window_size
    self.current_step = window_size
    self.balance = initial_balance
    self.positions = []
    self.volume = volume
    self.min_hold_time = 10
    self.action_space = Discrete(3)
    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.window_size, 14), dtype=np.float32)

  def step(self, action):
    current_price = self.data['close'].iloc[self.current_step]
    reward = 0
    realized_reward = 0
    holding_reward = 0

    current_drawdown = (self.initial_balance - self.balance) / self.initial_balance
    if current_drawdown > self.max_drawdown_pct:
      state = self.data.iloc[self.current_step - self.window_size:self.current_step][
        ['EMA_12', 'EMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'ADX',
        'ADX_Positive', 'ADX_Negative', 'open', 'high', 'low', 'close', 'tick_volume']
      ].values
      return state, reward, True, False, {}

    if action == 1 and self.balance >= current_price * self.volume:
      realized_reward = self._close_positions("sell", current_price)
      self.positions.append(["buy", current_price, self.current_step])
      self.balance -= current_price * self.volume

    elif action == 2 and self.balance >= current_price * self.volume:
      realized_reward = self._close_positions("buy", current_price)
      self.positions.append(["sell", current_price, self.current_step])
      self.balance -= current_price * self.volume

    if self.positions:
      unrealized_profit = sum(
        ((current_price - p[1]) * self.volume if p[0] == "buy" 
          else (p[1] - current_price) * self.volume)
        for p in self.positions
      )
      position_value = sum(p[1] * self.volume for p in self.positions)
      holding_reward = np.clip(unrealized_profit / (position_value * 0.1), -1, 1)

    reward = realized_reward + holding_reward

    self.current_step += 1
    done = self.current_step >= len(self.data) - 1
    truncated = False
    state = self.data.iloc[self.current_step - self.window_size:self.current_step][
      ['EMA_12', 'EMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'ADX',
      'ADX_Positive', 'ADX_Negative', 'open', 'high', 'low', 'close', 'tick_volume']
    ].values

    return state, reward, done, truncated, {}

  def reset(self, seed=None, **kwargs):
    if seed is not None:
      np.random.seed(seed)  # Optional: Set the seed for randomness

    self.current_step = self.window_size
    self.balance = 10000
    self.positions = []
    return self.data.iloc[self.current_step - self.window_size:self.current_step].values, {}

  def _close_positions(self, position_type, current_price):
    total_profit = 0
    new_positions = []
    for position in self.positions:
      if position[0] == position_type:
        entry_price = position[1]
        cost_basis = entry_price * self.volume
        profit = (entry_price - current_price) * self.volume if position_type == "buy" else (current_price - entry_price) * self.volume
        self.balance += cost_basis + profit
        total_profit += profit
      else:
        new_positions.append(position)
    self.positions = new_positions
    return total_profit

  
  
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
