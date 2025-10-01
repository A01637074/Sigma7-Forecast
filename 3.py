%pip install alpaca-trade-api
import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from alpaca_trade_api.rest import REST, TimeFrame

# ==================== CONFIGURACIÓN ALPACA ====================
API_KEY = "PK6PR8K71SUI4RGI2JMB"        # <--- Pon aquí tu API_KEY
API_SECRET = "2t6mK57kzfZNFJKVKIAAEI7Grvj0rOrSajwnVXvq"     # <--- Pon aquí tu API_SECRET
BASE_URL = "https://paper-api.alpaca.markets"   # Paper trading

def get_alpaca_data(ticker: str, start_date: str, end_date: str, timeframe: str = "1Day") -> pd.DataFrame:
    """
    Descarga datos históricos de Alpaca y los devuelve en formato similar a yfinance.
    """
    alpaca = REST(API_KEY, API_SECRET, base_url=BASE_URL)

    # Selección de intervalo
    if timeframe == "1Day":
        tf = TimeFrame.Day
    elif timeframe == "1Hour":
        tf = TimeFrame.Hour
    else:
        raise ValueError("Solo se soporta '1Day' o '1Hour' en este ejemplo")

    # Obtención de barras
    bars = alpaca.get_bars(ticker, tf, start=start_date, end=end_date).df

    if bars.empty:
        raise ValueError(f"Sin datos de Alpaca para {ticker} entre {start_date} y {end_date}")

    # Alpaca returns a multi-index dataframe when requesting multiple tickers,
    # but a simple dataframe when requesting a single ticker.
    # We handle both cases by checking if 'symbol' is in the columns.
    if 'symbol' in bars.columns:
        bars = bars[bars["symbol"] == ticker].set_index("timestamp")
    else:
        # If 'symbol' is not in columns, it's likely a single ticker request,
        # and 'timestamp' is already the index or a regular column.
        # Ensure 'timestamp' is the index.
        if 'timestamp' in bars.columns:
            bars = bars.set_index("timestamp")
        # If 'timestamp' is already the index, do nothing.


    # Renombrar columnas para parecerse a yfinance
    bars = bars.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    # No existe "Adj Close" en Alpaca, usamos Close
    bars["Adj Close"] = bars["Close"]

    return bars


# ==================== PARÁMETROS GLOBALES ====================
TICKERS = ["AAPL", "NVDA"]          # <- edita tu canasta
START_DATE = "2021-01-01"
END_DATE   = "2024-12-31"
FORECAST_HORIZONS = [1, 5]          # 1 día y ~1 semana
# ---- Trading (Paper) ----
TRADE_ENABLED = False               # <- pon True para operar con Alpaca
LONG_ONLY = True                    # True=cerrar en 0 al vender; False=permite short
TARGET_QTY = 1                      # tamaño deseado por ticker
# ---- Backtest ----
RUN_BACKTEST = False                # pon True si quieres ver un sanity-check simple


# ==================== INDICADORES ====================

def EMA(series: pd.Series, period: int) -> pd.Series:
    """Media exponencial clásica."""
    return series.ewm(span=period, adjust=False).mean()

def calcular_MACD(close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = EMA(close_series, fast)
    ema_slow = EMA(close_series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    macd_slope = macd_line.diff()
    return macd_line, signal_line, histogram, macd_slope

def calcular_RSI_Wilder(close_series: pd.Series, period: int = 14) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ==================== FORECASTING ====================

def _forecast_last_step(series: pd.Series, horizons: List[int]) -> Dict[int, float]:
    series = series.dropna()
    if len(series) < 25:
        last = float(series.iloc[-1]) if len(series) else np.nan
        return {h: last for h in horizons}
    try:
        max_h = max(horizons)
        fit = ARIMA(series, order=(1, 1, 1)).fit()
        fc = fit.get_forecast(steps=max_h)
        mean_fc = pd.Series(fc.predicted_mean, index=range(1, max_h + 1))
        return {h: float(mean_fc.loc[h]) for h in horizons}
    except Exception:
        last = float(series.iloc[-1]) if len(series) else np.nan
        return {h: last for h in horizons}


# ==================== EVALUACIÓN MULTIHORIZONTE ====================

def evaluar_indicadores_multi_horizon(ticker: str, start_date: str, end_date: str,
                                      forecast_horizons: List[int]) -> Dict:
    """
    Calcula MACD/RSI actuales y su forecast por horizonte.
    Devuelve estados y recomendación (BUY/SELL/HOLD) por horizonte.
    """
    data = get_alpaca_data(ticker, start_date, end_date, timeframe="1Day")
    if data.empty:
        raise ValueError(f"Sin datos para {ticker} en {start_date}..{end_date}")

    close = data["Adj Close"].dropna()

    macd_line, signal_line, hist, macd_slope = calcular_MACD(close)
    rsi = calcular_RSI_Wilder(close)

    macd_now = float(np.nan_to_num(macd_line.iloc[-1], nan=0.0))
    signal_now = float(np.nan_to_num(signal_line.iloc[-1], nan=0.0))
    macd_slope_now = float(np.nan_to_num(macd_slope.iloc[-1], nan=0.0))
    rsi_now = float(np.nan_to_num(rsi.iloc[-1], nan=50.0))

    macd_forecasts = _forecast_last_step(macd_line, forecast_horizons)
    rsi_forecasts  = _forecast_last_step(rsi,       forecast_horizons)

    results = {
        "ticker": ticker,
        "macd_now": macd_now,
        "signal_now": signal_now,
        "macd_slope_now": macd_slope_now,
        "rsi_now": rsi_now,
    }

    for h in forecast_horizons:
        macd_f = float(macd_forecasts[h])
        rsi_f  = float(rsi_forecasts[h])

        # Estado MACD
        if macd_now > signal_now and macd_slope_now > 0 and macd_f > macd_now:
            macd_state = "bullish"
        elif macd_now < signal_now and macd_slope_now < 0 and macd_f < macd_now:
            macd_state = "bearish"
        else:
            macd_state = "neutral"

        # Estado RSI
        if rsi_now >= 70 or rsi_f >= 70:
            rsi_state = "overbought"
        elif rsi_now <= 30 or rsi_f <= 30:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

        # Regla de decisión
        if macd_state == "bullish" and rsi_state != "overbought":
            trend = "bullish"
            recommendation = "BUY"
        elif macd_state == "bearish" and rsi_state != "oversold":
            trend = "bearish"
            recommendation = "SELL"
        else:
            trend = "neutral"
            recommendation = "HOLD"

        results[f"horizon_{h}_days"] = {
            "macd_forecast": macd_f,
            "rsi_forecast": rsi_f,
            "macd_state": macd_state,
            "rsi_state": rsi_state,
            "trend": trend,
            "recommendation": recommendation,
        }

    return results


# ==================== EJECUTOR ALPACA (TRADING) ====================

def _alpaca_client():
    if not (API_KEY and API_SECRET):
        raise RuntimeError("Faltan credenciales Alpaca")
    return REST(API_KEY, API_SECRET, base_url=BASE_URL)

def get_position_qty(alpaca, symbol: str) -> int:
    try:
        p = alpaca.get_position(symbol)
        return int(float(p.qty))
    except Exception:
        return 0

def place_market_order(alpaca, symbol: str, qty: int):
    side = "buy" if qty > 0 else "sell"
    alpaca.submit_order(symbol=symbol, qty=abs(qty), side=side,
                        type="market", time_in_force="day")

def trade_from_signal(alpaca, symbol: str, signal: str, target_qty: int = 1, long_only: bool = True):
    current = get_position_qty(alpaca, symbol)
    if signal == "BUY":
        target = max(target_qty, 0)
        if current < target:
            place_market_order(alpaca, symbol, target - current)
            print(f"[{symbol}] BUY → posición {current}→{target}")
        else:
            print(f"[{symbol}] BUY (ya en {current})")
    elif signal == "SELL":
        if long_only:
            if current > 0:
                place_market_order(alpaca, symbol, -current)
                print(f"[{symbol}] SELL (close) → posición {current}→0")
            else:
                print(f"[{symbol}] SELL (sin posición long)")
        else:
            target = -max(target_qty, 0)
            if current > target:
                place_market_order(alpaca, symbol, target - current)
                print(f"[{symbol}] SELL → posición {current}→{target}")
            else:
                print(f"[{symbol}] SELL (ya en {current})")
    else:
        print(f"[{symbol}] HOLD (posición {current})")


def run_signals_and_trade(tickers: List[str], start_date: str, end_date: str,
                          horizon: int = 1, target_qty: int = 1, long_only: bool = True):
    alpaca = _alpaca_client()
    for t in tickers:
        res = evaluar_indicadores_multi_horizon(t, start_date, end_date, [horizon])
        sig = res[f"horizon_{horizon}_days"]["recommendation"]
        trade_from_signal(alpaca, t, sig, target_qty=target_qty, long_only=long_only)


# ==================== MINI-BACKTEST ====================

def backtest_signals(ticker: str, start_date: str, end_date: str, horizon: int = 1) -> Tuple[Dict, pd.Series]:
    data = get_alpaca_data(ticker, start_date, end_date, timeframe="1Day")
    close = data["Adj Close"].dropna()

    recs = []
    idx = close.index
    for i in range(60, len(close)):
        sub = close.iloc[:i]
        res = evaluar_indicadores_multi_horizon(
            ticker, start_date=str(sub.index[0].date()),
            end_date=str(sub.index[-1].date()),
            forecast_horizons=[horizon]
        )
        rec = res[f"horizon_{horizon}_days"]["recommendation"]
        recs.append((idx[i], rec))

    sig = pd.Series({d: (1 if r=="BUY" else (-1 if r=="SELL" else 0)) for d, r in recs}).reindex(close.index).fillna(0)
    ret = close.pct_change(horizon).shift(-horizon).reindex(sig.index)
    pnl = sig * ret
    equity = (1 + pnl.fillna(0)).cumprod()

    summary = {
        "CAGR_est": (equity.iloc[-1] ** (252/len(equity)) - 1) if len(equity) > 252 else np.nan,
        "Sharpe_est": (pnl.mean() / pnl.std() * np.sqrt(252)) if pd.notna(pnl.std()) and pnl.std() != 0 else np.nan,
        "HitRate": float((pnl > 0).mean()),
        "Trades_aprox": int((sig.diff().abs() > 0).sum() / 2),
    }
    return summary, equity


# ==================== MAIN ====================

def main():
    resultados = []
    for t in TICKERS:
        r = evaluar_indicadores_multi_horizon(t, START_DATE, END_DATE, FORECAST_HORIZONS)
        resultados.append(r)

    for r in resultados:
        print(f"Ticker: {r['ticker']}")
        print(f"  MACD ahora: {r['macd_now']:.4f}  Señal: {r['signal_now']:.4f}  RSI: {r['rsi_now']:.2f}")
        for h in FORECAST_HORIZONS:
            k = f"horizon_{h}_days"
            x = r[k]
            print(f"  Horizonte {h}d → MACDf: {x['macd_forecast']:.4f}  RSIf: {x['rsi_forecast']:.2f}  "
                  f"MACD: {x['macd_state']}  RSI: {x['rsi_state']}  "
                  f"{x['trend']}  → {x['recommendation']}")
        print("-" * 60)

    if RUN_BACKTEST:
        print("\n=== Backtest rápido (h=1) ===")
        for t in TICKERS:
            summ, _eq = backtest_signals(t, START_DATE, END_DATE, horizon=1)
            print(f"{t}: {summ}")

    if TRADE_ENABLED:
        print("\n== Enviando órdenes a Alpaca (PAPER) ==")
        h = FORECAST_HORIZONS[0]
        run_signals_and_trade(TICKERS, START_DATE, END_DATE, horizon=h,
                              target_qty=TARGET_QTY, long_only=LONG_ONLY)
    else:
        print("\n[INFO] TRADE_ENABLED=False → no se enviaron órdenes. "
              "Cambia a True para operar en Paper Trading.")


if __name__ == "__main__":
    main()