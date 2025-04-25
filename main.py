import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import requests
import os
import joblib

# Telegram credentials
TELEGRAM_CHAT_ID = '6234582096'
TELEGRAM_BOT_TOKEN = '8007317168:AAHH9MjC4ScL9ZF9K9QN81VloIxVd9arNJQ'

# Paths
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'alphaedge_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
DATA_PATH = 'data/alphaedge_data.csv'
ACCURACY_LOG_PATH = 'accuracy/model_accuracy_log.csv'
TICKER = 'RELIANCE.NS'

# Telegram alert
def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"[ERROR] Telegram failed: {e}")

# Fetch stock data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.dropna(inplace=True)
    return df

# Apply technical indicators
def apply_indicators(df):
    df = df.copy()
    df['EMA10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    bb = BollingerBands(close=df['Close'])
    df['BB_H'] = bb.bollinger_hband()
    df['BB_L'] = bb.bollinger_lband()
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['STOCH'] = stoch.stoch()
    return df.dropna()

# Feature/target preparation
def prepare_data(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    features = ['EMA10', 'EMA20', 'RSI', 'MACD', 'BB_H', 'BB_L', 'STOCH']
    return df[features], df['Target']

# Data normalization
def normalize_data(X, fit_scaler=False):
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    return X_scaled

# Log accuracy
def log_accuracy(date, accuracy):
    os.makedirs(os.path.dirname(ACCURACY_LOG_PATH), exist_ok=True)
    if not os.path.exists(ACCURACY_LOG_PATH):
        with open(ACCURACY_LOG_PATH, "w") as f:
            f.write("Date,Accuracy\n")
    with open(ACCURACY_LOG_PATH, "a") as f:
        f.write(f"{date},{accuracy:.4f}\n")

# Initial training
def initial_training():
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=5 * 365)
    df = fetch_data(TICKER, start=start, end=end)
    df = apply_indicators(df)
    X, y = prepare_data(df)
    X_scaled = normalize_data(X, fit_scaler=True)

    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    model.partial_fit(X_scaled, y, classes=np.array([0, 1]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    timestamped_path = os.path.join(MODEL_DIR, f"alphaedge_model_{datetime.datetime.now().date()}.pkl")
    joblib.dump(model, timestamped_path)

    df_to_save = df[X.columns.tolist() + ['Target']].copy()
    df_to_save['Date'] = df_to_save.index
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df_to_save.to_csv(DATA_PATH, index=False)
    return model

# Incremental daily learning
def incremental_update():
    model = joblib.load(MODEL_PATH)
    past_df = pd.read_csv(DATA_PATH)
    past_df['Date'] = pd.to_datetime(past_df['Date'])

    last_date = past_df['Date'].max().date()
    today = datetime.datetime.now().date()

    if last_date < today:
        new_start = last_date + datetime.timedelta(days=1)
        new_end = datetime.datetime.now()
        new_df = fetch_data(TICKER, new_start, new_end)

        if not new_df.empty:
            new_df = apply_indicators(new_df)
            X_new, y_new = prepare_data(new_df)

            if not X_new.empty:
                X_new_scaled = normalize_data(X_new, fit_scaler=False)
                model.partial_fit(X_new_scaled, y_new)

                joblib.dump(model, MODEL_PATH)
                timestamped_path = os.path.join(MODEL_DIR, f"alphaedge_model_{today}.pkl")
                joblib.dump(model, timestamped_path)

                new_df_to_save = new_df[X_new.columns.tolist() + ['Target']].copy()
                new_df_to_save['Date'] = new_df_to_save.index
                new_df_to_save.to_csv(DATA_PATH, mode='a', header=False, index=False)

                print(f"✅ Model updated with {len(X_new)} new row(s).")
            else:
                print("⚠️ New data is incomplete — indicators not ready.")
        else:
            print("⚠️ No new rows returned by Yahoo Finance.")
    else:
        print("✅ Model already up to date with the latest data.")

    # Accuracy check
    recent_df = past_df.tail(100).copy()
    feature_cols = ['EMA10', 'EMA20', 'RSI', 'MACD', 'BB_H', 'BB_L', 'STOCH']
    X_test = normalize_data(recent_df[feature_cols], fit_scaler=False)
    y_test = recent_df['Target']
    acc = accuracy_score(y_test, model.predict(X_test))
    log_accuracy(today.strftime("%Y-%m-%d"), acc)

    # Signal prediction
    X_latest = X_test[[-1]]
    prediction = model.predict(X_latest)[0]
    signal = "BUY" if prediction == 1 else "SELL"
    price = fetch_data(TICKER, today - datetime.timedelta(days=5), today).iloc[-1]['Close']
    qty = int(100000 / price)

    report = (
        f"\U0001F4C8 Hello Iswar\n\nAlphaEdge Daily Report\n"
        f"\U0001F4C5 Date: {today}\n"
        f"\U0001F4CA Stock: {TICKER}\n"
        f"\U0001F4E2 Signal: {signal}\n"
        f"\U0001F4B0 Price: ₹{price:.2f} | Qty: {qty}\n"
        f"\U0001F9E0 Accuracy (last 100): {acc * 100:.2f}%\n"
        f"\U0001F4DD Model updated"
    )

    print(report)
    send_telegram_message(report)

# Main
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Initial training...")
        initial_training()
    else:
        print("Performing daily update...")
        incremental_update()
