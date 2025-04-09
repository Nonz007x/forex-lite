import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from pathlib import Path
from datetime import datetime
from prototype_3 import StockTradingEnv

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
  positions = []
  return StockTradingEnv(df[:149002], features=14, positions=positions)
  
if __name__ == '__main__':

  print("loading file...")
  parent_dir = Path(__file__).parent.parent.parent
  csv_path = parent_dir / "train-data" / "eurusd.csv"
  data = load_data_from_csv(csv_path)

  print("creating indicators...")
  df = create_indicator(data)

  print("creating environment...")
  env = SubprocVecEnv([make_env for _ in range(12)])
  # env = DummyVecEnv([make_env])
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

  print("start training...")
  model.learn(total_timesteps=122880)
  print("finished training", datetime.now())
  script_dir = Path(__file__).parent
  file_path = script_dir / "model"
  model.save(str(file_path))
  
  # current_dir = Path(__file__).parent
  # model_path = current_dir / "model"
  # model = PPO.load(model_path)

  # env = DummyVecEnv([lambda: StockTradingEnv(df[-61:], 14)])
  # done = False
  # obs = env.reset()

  # action_signal, _ = model.predict(obs)
  # obs, reward, done, infos = env.step(action_signal)

  # for i, info in enumerate(infos):
  #   print(f"[ENV {i}] Action: {info.get('action', 'N/A')}, Reward: {reward[i]}, Done: {done[i]}")
