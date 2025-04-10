import sys
import zmq
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from models.prototype_3.prototype_3 import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

context = zmq.Context()
socket = context.socket(zmq.ROUTER)

socket.bind("tcp://*:5557")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("ZeroMQ server is running...")

MODEL_CACHE = {}
ENV_CACHE = {}

def load_model(model_path):
  """
  Load model only once and cache it
  """
  # Convert to string path for consistent dictionary key
  model_path_str = str(model_path)
  
  if model_path_str not in MODEL_CACHE:
    try:
      MODEL_CACHE[model_path_str] = PPO.load(model_path)
      print(f"Model loaded from {model_path}")
    except Exception as e:
      print(f"Error loading model from {model_path}: {e}")
      return

  return MODEL_CACHE[model_path_str]

def preprocess_data(data_list):
  df = pd.DataFrame(data_list)
  df = df.iloc[::-1]
  df.drop(columns=['time'], errors='ignore', inplace=True)
  df[['open', 'high', 'low', 'close', 'tick_volume']] = df[[
    'open', 'high', 'low', 'close', 'tick_volume']].apply(pd.to_numeric, errors='coerce')
  
  df['EMA_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
  df['EMA_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
  df['MACD'] = MACD(df['close']).macd()
  df['RSI'] = RSIIndicator(df['close']).rsi()
  bb = BollingerBands(df['close'], window=20)
  df['BB_Upper'] = bb.bollinger_hband()
  df['BB_Lower'] = bb.bollinger_lband()
  adx = ADXIndicator(df['high'], df['low'], df['close'])
  df['ADX'] = adx.adx()
  df['ADX_Positive'] = adx.adx_pos()
  df['ADX_Negative'] = adx.adx_neg()

  df.dropna(inplace=True)

  return df

def run_prediction(model_id, data, positions):
  current_dir = Path(__file__).parent
  model_path = current_dir / "models" / f"{model_id}" / "model"
  model = load_model(model_path)

  session_key = f"env_{model_id}"

  df = preprocess_data(data)

  if session_key not in ENV_CACHE:
    ENV_CACHE[session_key] = DummyVecEnv([
      lambda: StockTradingEnv(data=df, features=14, positions=positions)
    ])
  else:
    env = ENV_CACHE[session_key]
    env.envs[0].update_data(df)  # ← Update with latest data

  env = ENV_CACHE[session_key]

  obs = env.envs[0]._get_state()
  obs = np.expand_dims(obs, axis=0)

  action_signal, _ = model.predict(obs)
  _, _, _, infos = env.step(action_signal)

  print("Balance: " + f"{infos[0].get('balance')}")
  return json.dumps({"action": infos[0].get("action")}).encode("utf-8")

while True:
  try:
    events = dict(poller.poll(timeout=100))

    if socket in events:
      identity, message = socket.recv_multipart()
      message_str = message.decode("utf-8", "ignore")
      
      if not message_str.strip():
        continue

      print(f"Received from {identity.hex()}")
      response = None

      if "backtest" in message_str:
        try:
          data = json.loads(message_str)
          if isinstance(data, str):
            data = json.loads(data)
        except json.JSONDecodeError as e:
          print(f"JSON Decode Error: {e} | Raw: {message_str}")
          data = None

        if not data:
          response = b"No data found!"

        model_id = data.get("model_id")
        market_data = data.get("market_data")
        positions = data.get("positions")
        market_data = json.loads(market_data)
        response = run_prediction(model_id, market_data, positions)

      elif "init" in message_str:
        response = b"Connection established..."

      elif "end" in message_str:
        response = b"Closing connection..."

      if response:
        socket.send_multipart([identity, response])
        print(f"Sent to {identity.hex()}: {response}")

    time.sleep(0.001)

  except KeyboardInterrupt:
    print("Server shutting down...")
    break
