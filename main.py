import os
import time
import json
import logging
import threading
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import requests

try:
    from groq import Groq
except ImportError:
    raise ImportError("GROQ package not installed. Run: pip install groq")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

SCAN_INTERVAL_SEC = 15 * 60
TRADE_CHECK_SEC = 60
COIN_DELAY_SEC = 4
RETRY_AFTER_429_SEC = 65

TP_PCT = 3.0
SL_PCT = 1.5
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
RSI_PERIOD = 14
MIN_CONFIDENCE = 65

GROQ_MODEL = "mixtral-8x7b-32768"

COINS = [
    "bitcoin",
    "ethereum",
    "binancecoin",
    "solana",
    "ripple",
    "cardano",
    "avalanche-2",
    "polkadot",
    "chainlink",
    "dogecoin",
    "shiba-inu",
    "matic-network",
    "litecoin",
    "uniswap",
    "cosmos",
    "stellar",
    "monero",
    "tron",
    "near-protocol",
    "the-open-network",
]

COIN_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "solana": "SOL",
    "ripple": "XRP",
    "cardano": "ADA",
    "avalanche-2": "AVAX",
    "polkadot": "DOT",
    "chainlink": "LINK",
    "dogecoin": "DOGE",
    "shiba-inu": "SHIB",
    "matic-network": "MATIC",
    "litecoin": "LTC",
    "uniswap": "UNI",
    "cosmos": "ATOM",
    "stellar": "XLM",
    "monero": "XMR",
    "tron": "TRX",
    "near-protocol": "NEAR",
    "the-open-network": "TON",
}

GROQ_API_KEY = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
NEWS_API_KEY = ""
COINGECKO_API_KEY = ""
groq_client = None

active_trades: Dict[str, Dict] = {}
trades_lock = threading.Lock()

def _get(url: str, params: dict = None, headers: dict = None, retries: int = 3) -> Optional[dict]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)

            if resp.status_code == 429:
                log.warning("[RATE_LIMIT] 429 - sleeping %ds (attempt %d/%d)",
                           RETRY_AFTER_429_SEC, attempt, retries)
                time.sleep(RETRY_AFTER_429_SEC)
                continue

            if resp.status_code == 404:
                log.warning("[NOT_FOUND] 404 - %s", url)
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            log.error("[TIMEOUT] %s attempt %d/%d", url, attempt, retries)
            if attempt < retries:
                time.sleep(6)
        except requests.exceptions.JSONDecodeError:
            log.error("[JSON_ERROR] Empty/invalid JSON from %s", url)
            return None
        except requests.exceptions.RequestException as exc:
            log.error("[HTTP_ERROR] %s attempt %d/%d - %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(6)
    return None

def cg_headers() -> dict:
    h = {"accept": "application/json"}
    if COINGECKO_API_KEY:
        h["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return h

def fetch_current_price(coin_id: str) -> Optional[float]:
    data = _get(
        f"{COINGECKO_BASE}/simple/price",
        params={"ids": coin_id, "vs_currencies": "usd"},
        headers=cg_headers(),
    )
    if data and coin_id in data:
        return float(data[coin_id]["usd"])
    return None

def fetch_ohlcv_hourly(coin_id: str) -> Optional[List[float]]:
    data = _get(
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": "3"},
        headers=cg_headers(),
    )
    if not data or "prices" not in data:
        return None
    prices = data["prices"]
    if len(prices) < RSI_PERIOD + 2:
        return None
    return [float(p[1]) for p in prices]

def compute_rsi(closes: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    if len(closes) < period + 2:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1 + rs)), 2)

def fetch_news_headlines(query: str, max_articles: int = 5) -> List[str]:
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": NEWS_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        log.error("[NEWS_ERROR] %s", exc)
        return []

def ask_groq(coin_id: str, symbol: str, price: float, rsi: float, headlines: List[str]) -> Optional[Dict]:
    news_block = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news available."
    
    prompt = f"""You are a professional crypto trading analyst. Analyze the data below and respond with a trading signal.

Coin: {symbol} ({coin_id})
Current Price: ${price:,.6f}
RSI(14): {rsi}

Recent News Headlines:
{news_block}

Decision rules:
- BUY → RSI < 40 (oversold) AND news is neutral or positive
- SELL → RSI > 60 (overbought) AND news is neutral or negative
- NO TRADE → uncertain or conflicting signals

Respond ONLY with a single valid JSON object. No markdown, no explanation, no extra text.
Example format: {{"action": "BUY", "confidence": 75, "reason": "RSI oversold with positive sentiment"}}

Constraints:
- "action" must be exactly one of: BUY, SELL, NO TRADE
- "confidence" must be an integer between 0 and 100
- "reason" must be one short sentence"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a crypto trading analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=200,
            timeout=30,
        )
        
        text = chat_completion.choices[0].message.content.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            result = json.loads(text)

        action = str(result.get("action", "")).upper().strip()
        if action not in ("BUY", "SELL", "NO TRADE"):
            log.warning("[GROQ] Unexpected action '%s' for %s", action, symbol)
            return {"action": "NO TRADE", "confidence": 0, "reason": "Invalid Groq response"}

        return {
            "action": action,
            "confidence": int(result.get("confidence", 0)),
            "reason": str(result.get("reason", "N/A"))[:200],
        }

    except json.JSONDecodeError as e:
        log.error("[GROQ_PARSE] JSON decode failed: %s", e)
        return None
    except Exception as exc:
        log.error("[GROQ_ERROR] %s: %s", type(exc).__name__, exc)
        return None

def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        log.error("[TELEGRAM_ERROR] %s", exc)
        return False

def fmt_signal_msg(action: str, symbol: str, coin_id: str, price: float, rsi: float, confidence: int, reason: str) -> str:
    emoji = "🟢" if action == "BUY" else "🔴"
    tp = price * (1 + TP_PCT / 100)
    sl = price * (1 - SL_PCT / 100)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{emoji} <b>{action} SIGNAL - {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry: <b>${price:,.6f}</b>\n"
        f"🎯 TP: <b>${tp:,.6f}</b> (+{TP_PCT}%)\n"
        f"🛑 SL: <b>${sl:,.6f}</b> (-{SL_PCT}%)\n"
        f"📊 RSI: {rsi}\n"
        f"🤖 Confidence: {confidence}%\n"
        f"💡 {reason}\n"
        f"🕐 {now}"
    )

def fmt_tp_msg(symbol: str, entry: float, exit_price: float, pct: float) -> str:
    return (
        f"✅ <b>TAKE PROFIT - {symbol}</b>\n"
        f"Entry: ${entry:,.6f}\n"
        f"Exit: ${exit_price:,.6f}\n"
        f"Gain: +{pct:.2f}% 🎉"
    )

def fmt_sl_msg(symbol: str, entry: float, exit_price: float, pct: float) -> str:
    return (
        f"❌ <b>STOP LOSS - {symbol}</b>\n"
        f"Entry: ${entry:,.6f}\n"
        f"Exit: ${exit_price:,.6f}\n"
        f"Loss: -{abs(pct):.2f}%"
    )

def open_trade(coin_id: str, symbol: str, action: str, entry_price: float) -> None:
    tp = entry_price * (1 + TP_PCT / 100)
    sl = entry_price * (1 - SL_PCT / 100)
    with trades_lock:
        active_trades[coin_id] = {
            "symbol": symbol,
            "action": action,
            "entry": entry_price,
            "tp": tp,
            "sl": sl,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
    log.info("[TRADE_OPEN] %s %s @ $%.6f", action, symbol, entry_price)

def cleanup_old_trades(max_age_hours: int = 24) -> None:
    with trades_lock:
        now = datetime.now(timezone.utc)
        to_remove = []
        for coin_id, trade in active_trades.items():
            try:
                opened_at = datetime.fromisoformat(trade["opened_at"])
                age_hours = (now - opened_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_remove.append(coin_id)
            except (KeyError, ValueError):
                to_remove.append(coin_id)
        for coin_id in to_remove:
            log.warning("[CLEANUP] Removing stale trade: %s", coin_id)
            active_trades.pop(coin_id, None)

def check_trades() -> None:
    cleanup_old_trades()
    
    with trades_lock:
        snapshot = dict(active_trades)

    for coin_id, trade in snapshot.items():
        try:
            price = fetch_current_price(coin_id)
            if price is None:
                continue

            symbol = trade["symbol"]
            entry = trade["entry"]
            tp = trade["tp"]
            sl = trade["sl"]
            action = trade["action"]

            hit_tp = (action == "BUY" and price >= tp) or (action == "SELL" and price <= tp)
            hit_sl = (action == "BUY" and price <= sl) or (action == "SELL" and price >= sl)

            if hit_tp:
                pct = (price - entry) / entry * 100 if action == "BUY" else (entry - price) / entry * 100
                send_telegram(fmt_tp_msg(symbol, entry, price, pct))
                log.info("[TP] %s | +%.2f%%", symbol, pct)
                with trades_lock:
                    active_trades.pop(coin_id, None)
            elif hit_sl:
                pct = (entry - price) / entry * 100 if action == "BUY" else (price - entry) / entry * 100
                send_telegram(fmt_sl_msg(symbol, entry, price, pct))
                log.info("[SL] %s | -%.2f%%", symbol, abs(pct))
                with trades_lock:
                    active_trades.pop(coin_id, None)
        except Exception as exc:
            log.error("[CHECK_TRADE_ERROR] %s - %s", coin_id, exc)

def trade_monitor_loop() -> None:
    log.info("[MONITOR] Started - checking every %ds", TRADE_CHECK_SEC)
    while True:
        try:
            check_trades()
        except Exception as exc:
            log.error("[MONITOR_ERROR] %s", exc)
        time.sleep(TRADE_CHECK_SEC)

def scan_coin(coin_id: str) -> None:
    symbol = COIN_SYMBOLS.get(coin_id, coin_id.upper())
    log.info("[SCAN] Checking %s (%s)...", symbol, coin_id)

    closes = fetch_ohlcv_hourly(coin_id)
    if not closes:
        log.info("[SKIP] %s - no market data", symbol)
        return

    rsi = compute_rsi(closes)
    if rsi is None:
        log.info("[SKIP] %s - insufficient candles (got %d)", symbol, len(closes))
        return

    log.info("[SCAN] %s | RSI=%.2f | Data points=%d", symbol, rsi, len(closes))

    if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
        log.info("[SKIP] %s - RSI=%.2f in neutral zone", symbol, rsi)
        return

    with trades_lock:
        if coin_id in active_trades:
            log.info("[SKIP] %s - active trade exists", symbol)
            return

    log.info("[TRIGGER] %s RSI=%.2f - fetching data...", symbol, rsi)

    price = fetch_current_price(coin_id)
    if price is None:
        log.warning("[SKIP] %s - price unavailable", symbol)
        return

    headlines = fetch_news_headlines(f"{symbol} crypto")
    log.info("[SCAN] %s - %d news headlines", symbol, len(headlines))

    ai_result = ask_groq(coin_id, symbol, price, rsi, headlines)
    if ai_result is None:
        log.warning("[SKIP] %s - Groq response invalid", symbol)
        return

    action = ai_result["action"]
    confidence = ai_result["confidence"]
    reason = ai_result["reason"]

    log.info("[SIGNAL] %s → %s | conf=%d%% | %s", symbol, action, confidence, reason)

    if action == "NO TRADE" or confidence < MIN_CONFIDENCE:
        log.info("[SKIP] %s - %s (conf=%d%%)", symbol, action, confidence)
        return

    msg = fmt_signal_msg(action, symbol, coin_id, price, rsi, confidence, reason)
    if send_telegram(msg):
        log.info("[SIGNAL] %s - Telegram sent ✓", symbol)

    open_trade(coin_id, symbol, action, price)

def scan_all_coins() -> None:
    log.info("═" * 55)
    log.info("STARTING SCAN - %d coins", len(COINS))
    
    signals = 0
    for idx, coin_id in enumerate(COINS, 1):
        try:
            with trades_lock:
                before = len(active_trades)
            scan_coin(coin_id)
            with trades_lock:
                after = len(active_trades)
            if after > before:
                signals += 1
        except Exception as exc:
            log.error("[SCAN_ERROR] %s - %s", coin_id, exc)
        
        if idx < len(COINS):
            time.sleep(COIN_DELAY_SEC)
    
    log.info("SCAN COMPLETE - Signals: %d | Active trades: %d", signals, len(active_trades))
    log.info("═" * 55)

def main() -> None:
    global GROQ_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    global NEWS_API_KEY, COINGECKO_API_KEY, groq_client

    required_vars = ["GROQ_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NEWS_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY = os.environ["NEWS_API_KEY"]
    COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "")

    groq_client = Groq(api_key=GROQ_API_KEY)

    log.info("=" * 55)
    log.info("CRYPTO SIGNAL BOT - GROQ VERSION")
    log.info(f"Model: {GROQ_MODEL}")
    log.info(f"Coins: {len(COINS)}")
    log.info(f"Interval: {SCAN_INTERVAL_SEC // 60} minutes")
    log.info(f"TP/SL: +{TP_PCT}% / -{SL_PCT}%")
    log.info(f"Min confidence: {MIN_CONFIDENCE}%")
    log.info("=" * 55)

    send_telegram(
        f"🤖 <b>Crypto Signal Bot Started (GROQ)</b>\n"
        f"Model: {GROQ_MODEL}\n"
        f"Scanning {len(COINS)} coins every {SCAN_INTERVAL_SEC // 60} min\n"
        f"TP +{TP_PCT}% | SL -{SL_PCT}% | Min conf {MIN_CONFIDENCE}%"
    )

    monitor_thread = threading.Thread(target=trade_monitor_loop, daemon=True, name="TradeMonitor")
    monitor_thread.start()

    while True:
        try:
            scan_all_coins()
        except Exception as exc:
            log.error("[MAIN_LOOP_ERROR] %s", exc)
        
        log.info("[WAIT] Next scan in %d minutes...", SCAN_INTERVAL_SEC // 60)
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
