
# - RSI (Wilder), MACD clásico
# - ARIMA(1,1,1) sobre indicadores (forecast multi-horizonte)
# - Reglas BUY/SELL/HOLD
# - Ejecutor opcional para Alpaca Paper Trading (seguro por defecto)
# - Mini-backtest opcional
# ------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

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
    """
    MACD:
      macd_line = EMA(fast) - EMA(slow)
      signal_line = EMA(macd_line, signal)
      histogram = macd_line - signal_line
      macd_slope = macd_line.diff()
    """
    ema_fast = EMA(close_series, fast)
    ema_slow = EMA(close_series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    macd_slope = macd_line.diff()
    return macd_line, signal_line, histogram, macd_slope

def calcular_RSI_Wilder(close_series: pd.Series, period: int = 14) -> pd.Series:
    """RSI con suavizado de Wilder (EWMA alpha=1/period)."""
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
    """
    Forecast del último paso para cada horizonte.
    Optimiza haciendo un get_forecast hasta max(horizons).
    En caso de error/serie corta, devuelve último valor observado.
    """
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
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if data.empty:
        raise ValueError(f"Sin datos para {ticker} en {start_date}..{end_date}")

    close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    close = close.dropna()

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


# ==================== EJECUTOR ALPACA (OPCIONAL) ====================

def _import_alpaca():
    try:
        from alpaca_trade_api import REST
        return REST
    except Exception as e:
        raise ImportError(
            "Necesitas instalar alpaca-trade-api: pip install alpaca-trade-api"
        ) from e

def _alpaca_client():
    REST = _import_alpaca()
    API_KEY = os.getenv("APCA_API_KEY_ID")
    API_SECRET = os.getenv("APCA_API_SECRET_KEY")
    BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    if not (API_KEY and API_SECRET):
        raise RuntimeError(
            "Faltan credenciales Alpaca. Exporta variables de entorno:\n"
            "  export APCA_API_KEY_ID=\"tu_key\"\n"
            "  export APCA_API_SECRET_KEY=\"tu_secret\"\n"
            "  export APCA_API_BASE_URL=\"https://paper-api.alpaca.markets\"  # opcional"
        )
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
    """
    signal ∈ {'BUY','SELL','HOLD'}
    long_only=True => al vender, se cierra a 0; False => permite ir a -target_qty
    """
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
                place_market_order(alpaca, symbol, -current)  # cerrar
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
    """Calcula señales y, si aplica, ejecuta órdenes."""
    alpaca = _alpaca_client()
    for t in tickers:
        res = evaluar_indicadores_multi_horizon(t, start_date, end_date, [horizon])
        sig = res[f"horizon_{horizon}_days"]["recommendation"]
        trade_from_signal(alpaca, t, sig, target_qty=target_qty, long_only=long_only)


# ==================== MINI-BACKTEST (OPCIONAL) ====================

def backtest_signals(ticker: str, start_date: str, end_date: str, horizon: int = 1) -> Tuple[Dict, pd.Series]:
    """
    Sanity-check simple: señal diaria => retorno futuro a 'horizon' días.
    Close-to-close, sin comisiones/slippage. Solo para referencia.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    close = close.dropna()

    recs = []
    idx = close.index
    # Warm-up para tener indicadores estables
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
    # 1) Señales e impresión
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

    # 2) Backtest opcional
    if RUN_BACKTEST:
        print("\n=== Backtest rápido (h=1) ===")
        for t in TICKERS:
            summ, _eq = backtest_signals(t, START_DATE, END_DATE, horizon=1)
            print(f"{t}: {summ}")

    # 3) Trading opcional (Paper)
    if TRADE_ENABLED:
        print("\n== Enviando órdenes a Alpaca (PAPER) ==")
        # por defecto operamos con el primer horizonte de la lista
        h = FORECAST_HORIZONS[0]
        run_signals_and_trade(TICKERS, START_DATE, END_DATE, horizon=h,
                              target_qty=TARGET_QTY, long_only=LONG_ONLY)
    else:
        print("\n[INFO] TRADE_ENABLED=False → no se enviaron órdenes. "
              "Cambia a True para operar en Paper Trading.")


if __name__ == "__main__":
    main()
