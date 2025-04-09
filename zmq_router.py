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

def evaluate_with_model(ohlc_json, positions, model):
  try:
    data_list = json.loads(ohlc_json)

    df = pd.DataFrame(data_list)
    df.drop(columns=['time'], errors='ignore', inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    if len(df) < 110:
      print(f"Error: Not enough data (only {len(df)} rows)")
      sys.exit(1)

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

    if len(df) < 61:
      print(f"Error: Invalid data shape after indicators {df.shape}")
      sys.exit(1)

    env = DummyVecEnv([lambda: StockTradingEnv(data=df, features=14, positions=positions)])

    obs = env.reset()
    action_signal, _ = model.predict(obs)
    obs, _, _, infos = env.step(action_signal)

    return infos[0].get("action")

  except Exception as e:
    print(f"ERROR: {e}", flush=True)

def run_prediction(model_id, data, positions):
  current_dir = Path(__file__).parent
  model_path = current_dir / "models" / f"{model_id}" / "model"
  model = load_model(model_path)
  action = evaluate_with_model(data, positions, model)
  prediction = json.dumps({"action": action})
  return bytes(prediction, "utf-8")

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

      if "signal_request" in message_str:
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

        response = run_prediction(model_id, market_data, positions)

      elif "backtest" in message_str:
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
