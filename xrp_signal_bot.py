import os
import json
import smtplib
import requests
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GMAIL_USER   = os.environ["GMAIL_USER"]
GMAIL_PASS   = os.environ["GMAIL_PASS"]
NOTIFY_EMAIL = os.environ["NOTIFY_EMAIL"]

# Top liquid coins on MEXC Futures (USDT-margined)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
    "DOTUSDT", "LTCUSDT", "SUIUSDT", "TONUSDT", "NEARUSDT",
]

INTERVAL = "30m"
LIMIT    = 100
BASE_URL = "https://api.mexc.com/api/v3"
STATE_FILE = "position_state.json"
SIGNAL_THRESHOLD = 3

# ─────────────────────────────────────────
# STATE MANAGEMENT — per-coin position tracking
# ─────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}   # { "XRPUSDT": {"position": "LONG", "entry_price": 1.05, "entry_time": "..."} }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_symbol_state(state, symbol):
    return state.get(symbol, {"position": None, "entry_price": None, "entry_time": None})

# ─────────────────────────────────────────
# STEP 1 — Fetch candles for a symbol
# ─────────────────────────────────────────
def fetch_candles(symbol):
    url = f"{BASE_URL}/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()

    if not raw or len(raw) < 50:
        raise ValueError(f"Not enough candle data for {symbol}")

    df = pd.DataFrame(raw)
    df = df.iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df

# ─────────────────────────────────────────
# STEP 2 — Indicators
# ─────────────────────────────────────────
def add_indicators(df):
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    ema12        = df["close"].ewm(span=12, adjust=False).mean()
    ema26        = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    sma20          = df["close"].rolling(20).mean()
    std20          = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > (df["vol_avg"] * 2)

    return df

# ─────────────────────────────────────────
# STEP 3 — Signal scoring
# ─────────────────────────────────────────
def detect_signal(df):
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    signals = []
    score   = 0

    if latest["rsi"] < 35:
        signals.append("✅ RSI oversold (<35)"); score += 2
    if latest["rsi"] > 50 and prev["rsi"] <= 50:
        signals.append("✅ RSI crossed above 50"); score += 1
    if latest["macd"] > latest["signal"] and prev["macd"] <= prev["signal"]:
        signals.append("✅ MACD bullish crossover"); score += 2
    if latest["close"] > latest["ema20"] > latest["ema50"]:
        signals.append("✅ Price above EMA20 & EMA50"); score += 2
    if latest["close"] <= latest["bb_lower"]:
        signals.append("✅ Price at lower Bollinger Band"); score += 2
    if latest["vol_spike"] and latest["close"] > prev["close"]:
        signals.append("✅ Volume spike on green candle"); score += 1

    if latest["rsi"] > 70:
        signals.append("🔴 RSI overbought (>70)"); score -= 2
    if latest["rsi"] < 50 and prev["rsi"] >= 50:
        signals.append("🔴 RSI crossed below 50"); score -= 1
    if latest["macd"] < latest["signal"] and prev["macd"] >= prev["signal"]:
        signals.append("🔴 MACD bearish crossover"); score -= 2
    if latest["close"] < latest["ema20"] < latest["ema50"]:
        signals.append("🔴 Price below EMA20 & EMA50"); score -= 2
    if latest["close"] >= latest["bb_upper"]:
        signals.append("🔴 Price at upper Bollinger Band"); score -= 2
    if latest["vol_spike"] and latest["close"] < prev["close"]:
        signals.append("🔴 Volume spike on red candle"); score -= 1

    bull_engulf = (prev["close"] < prev["open"] and latest["close"] > latest["open"] and
                   latest["open"] < prev["close"] and latest["close"] > prev["open"])
    bear_engulf = (prev["close"] > prev["open"] and latest["close"] < latest["open"] and
                   latest["open"] > prev["close"] and latest["close"] < prev["open"])
    if bull_engulf:
        signals.append("✅ Bullish Engulfing"); score += 2
    if bear_engulf:
        signals.append("🔴 Bearish Engulfing"); score -= 2

    return {
        "score": score, "signals": signals,
        "price": latest["close"], "atr": latest["atr"],
        "rsi": round(latest["rsi"], 2), "macd": round(latest["macd"], 6),
        "candle_time": latest["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
        "vol_spike": bool(latest["vol_spike"]),
    }

# ─────────────────────────────────────────
# STEP 4 — Decide action for one symbol
# ─────────────────────────────────────────
def decide_action(sig, sym_state):
    score    = sig["score"]
    position = sym_state["position"]

    new_direction = None
    if score >= SIGNAL_THRESHOLD:
        new_direction = "LONG"
    elif score <= -SIGNAL_THRESHOLD:
        new_direction = "SHORT"

    if position is None:
        if new_direction:
            return f"OPEN_{new_direction}", f"Naya {new_direction} signal (score {score})"
        return "WAIT", "Koi position nahi, signal weak"
    else:
        opposite = "SHORT" if position == "LONG" else "LONG"
        if new_direction == opposite:
            return f"CLOSE_{position}", f"Opposite signal aaya — {position} close karo"
        return "HOLD", f"{position} abhi valid hai"

def calc_risk_levels(direction, price, atr):
    if direction == "LONG":
        sl = round(price - atr, 6); tp = round(price + (atr * 2), 6)
    else:
        sl = round(price + atr, 6); tp = round(price - (atr * 2), 6)
    sl_pct = round(abs(price - sl) / price * 100, 2)
    tp_pct = round(abs(tp - price) / price * 100, 2)
    return sl, tp, sl_pct, tp_pct

# ─────────────────────────────────────────
# STEP 5 — Email (per coin, per action)
# ─────────────────────────────────────────
def send_email(symbol, action, sig, sym_state, sl=None, tp=None, sl_pct=None, tp_pct=None, pnl_pct=None):
    is_open   = action.startswith("OPEN_")
    direction = action.split("_")[1] if "_" in action else ""
    coin_name = symbol.replace("USDT", "")

    if is_open:
        emoji = "📈" if direction == "LONG" else "📉"
        subject = f"🚨 {coin_name} OPEN {direction} {emoji} | 30M | {sig['candle_time']}"
        bar_color = "#00c896" if direction == "LONG" else "#ff4d6d"
        order_side = "BUY (Open Long)" if direction == "LONG" else "SELL (Open Short)"
        levels_html = f"""
        <tr><td><b>Stop Loss</b></td><td>{sl} USDT (-{sl_pct}%)</td></tr>
        <tr><td><b>Take Profit</b></td><td>{tp} USDT (+{tp_pct}%)</td></tr>
        <tr><td><b>Risk:Reward</b></td><td>1 : 2 ✅</td></tr>
        """
        extra_steps = "4. Leverage: 10x-20x (apna risk dekh kar)<br>5. Set SL/TP as above<br>6. Max 2-5% portfolio per trade"
    else:
        pnl_color = "#00c896" if (pnl_pct or 0) >= 0 else "#ff4d6d"
        pnl_text  = f"{'+' if (pnl_pct or 0) >= 0 else ''}{pnl_pct}%"
        subject = f"🔔 {coin_name} CLOSE {direction} ✅ | PnL: {pnl_text} | {sig['candle_time']}"
        bar_color = "#ffaa00"
        order_side = "SELL (Close Long)" if direction == "LONG" else "BUY (Close Short)"
        entry_p = sym_state.get("entry_price", "N/A")
        levels_html = f"""
        <tr><td><b>Entry was</b></td><td>{entry_p} USDT</td></tr>
        <tr><td><b>Exit now</b></td><td>{sig['price']} USDT</td></tr>
        <tr><td><b>PnL</b></td><td style="color:{pnl_color};font-weight:bold;">{pnl_text}</td></tr>
        """
        extra_steps = "4. Existing position fully close karo"

    signals_html = "".join(f"<li>{s}</li>" for s in sig["signals"])

    body = f"""
    <html><body style="font-family:monospace;background:#0d0d0d;color:#e0e0e0;padding:24px;">
    <div style="max-width:600px;margin:auto;border:1px solid #222;border-radius:8px;overflow:hidden;">
      <div style="background:{bar_color};padding:16px 24px;">
        <h2 style="margin:0;color:#000;">⚡ {symbol} — {action.replace('_',' ')}</h2>
        <p style="margin:4px 0 0;color:#000;font-size:13px;">MEXC Futures · 30M · {sig['candle_time']}</p>
      </div>
      <div style="padding:24px;background:#111;">
        <h3 style="color:{bar_color};font-size:22px;margin:0 0 4px;">{direction}</h3>
        <p style="color:#aaa;margin:0 0 20px;">Score: {sig['score']}/10</p>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;width:140px;"><b>Current Price</b></td>
            <td style="color:#fff;">{sig['price']} USDT</td>
          </tr>
          {levels_html}
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;"><b>RSI (14)</b></td>
            <td style="color:#fff;">{sig['rsi']}</td>
          </tr>
          <tr>
            <td style="padding:8px 0;color:#aaa;"><b>MACD</b></td>
            <td style="color:#fff;">{sig['macd']}</td>
          </tr>
        </table>
        <div style="margin-top:20px;background:#1a1a1a;border-left:3px solid {bar_color};padding:12px 16px;border-radius:4px;">
          <p style="margin:0 0 8px;color:#aaa;font-size:12px;">Why this signal?</p>
          <ul style="margin:0;padding-left:18px;color:#ddd;font-size:13px;line-height:1.8;">{signals_html}</ul>
        </div>
        <div style="margin-top:20px;background:#1a1a1a;border-radius:4px;padding:12px 16px;">
          <p style="margin:0;color:#aaa;font-size:12px;">⚙️ MEXC Order</p>
          <p style="margin:8px 0 0;color:#ddd;font-size:13px;line-height:1.8;">
            1. MEXC → Futures → <b>{symbol}</b><br>
            2. Order Type: <b>Market Order</b><br>
            3. Action: <b>{order_side}</b><br>
            {extra_steps}
          </p>
        </div>
        <p style="margin-top:20px;color:#555;font-size:11px;border-top:1px solid #222;padding-top:12px;">
          ⚠️ Technical signal hai, guarantee nahi. Apna risk khud manage karo.
        </p>
      </div>
    </div>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = NOTIFY_EMAIL
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, NOTIFY_EMAIL, msg.as_string())

    print(f"   ✅ Email sent: {subject}")

# ─────────────────────────────────────────
# MAIN — loop over all symbols
# ─────────────────────────────────────────
def main():
    print(f"🔍 Scanning {len(SYMBOLS)} coins on MEXC Futures (30M)... [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
    state = load_state()
    emails_sent = 0

    for symbol in SYMBOLS:
        try:
            df  = fetch_candles(symbol)
            df  = add_indicators(df)
            sig = detect_signal(df)
            sym_state = get_symbol_state(state, symbol)

            action, reason = decide_action(sig, sym_state)
            print(f"📊 {symbol}: score={sig['score']} | pos={sym_state['position']} | action={action}")

            if action.startswith("OPEN_"):
                direction = action.split("_")[1]
                sl, tp, sl_pct, tp_pct = calc_risk_levels(direction, sig["price"], sig["atr"])
                send_email(symbol, action, sig, sym_state, sl=sl, tp=tp, sl_pct=sl_pct, tp_pct=tp_pct)
                state[symbol] = {
                    "position": direction,
                    "entry_price": sig["price"],
                    "entry_time": sig["candle_time"],
                }
                emails_sent += 1

            elif action.startswith("CLOSE_"):
                direction   = action.split("_")[1]
                entry_price = sym_state.get("entry_price", sig["price"])
                if direction == "LONG":
                    pnl_pct = round((sig["price"] - entry_price) / entry_price * 100, 2)
                else:
                    pnl_pct = round((entry_price - sig["price"]) / entry_price * 100, 2)
                send_email(symbol, action, sig, sym_state, pnl_pct=pnl_pct)
                state[symbol] = {"position": None, "entry_price": None, "entry_time": None}
                emails_sent += 1

            time.sleep(0.3)  # MEXC rate-limit ke liye chhota delay

        except Exception as e:
            print(f"   ⚠️ {symbol} skip — error: {e}")
            continue

    save_state(state)
    print(f"💾 State saved. Total emails sent this run: {emails_sent}")

if __name__ == "__main__":
    main()
