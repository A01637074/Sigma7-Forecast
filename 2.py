# ======================================================
# SIGMA7 — yfinance: MACD + RSI + ARIMA + Gráficas + Señales + Sim Trading
# ======================================================

import warnings, os, json, datetime as dt
from pathlib import Path
from typing import List, Dict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# ==================== PARÁMETROS ====================

TICKERS: List[str] = ["AAPL", "NVDA"]
START_DATE = "2021-01-01"
END_DATE   = dt.date.today().isoformat()

FORECAST_HORIZONS: List[int] = [1, 5]     # 1 día y ~1 semana
TRADE_HORIZON = 1                          # horizonte cuya señal seguimos (simulado)
TARGET_QTY = 1                             
LONG_ONLY = True                           # en SELL cierra a 0
OUTPUT_DIR = Path("outputs")

# ==================== UTILIDADES ====================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_close_from_yf(df: pd.DataFrame, sym: str | None = None) -> pd.Series:
    """
    Devuelve la serie de cierre (Adj Close si existe; si no, Close)
    para el símbolo indicado (o DataFrame de 1 símbolo).
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker
        cols = df[sym].columns.get_level_values(0).unique() if sym else []
        if "Adj Close" in df[sym].columns:
            s = df[sym]["Adj Close"]
        else:
            s = df[sym]["Close"]
    else:
        # Single ticker
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df["Close"]
    return s.dropna().astype(float)

def descargar_cierres_yf(tickers: List[str], start: str, end: str) -> Dict[str, pd.Series]:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,   # preferimos usar Adj Close explícito si está
        threads=True,
        group_by="ticker" if len(tickers) > 1 else "column"
    )
    out = {}
    if len(tickers) == 1:
        sym = tickers[0]
        out[sym] = get_close_from_yf(data)
    else:
        for sym in tickers:
            try:
                out[sym] = get_close_from_yf(data, sym)
            except Exception:
                out[sym] = pd.Series(dtype=float)
    return out

# ==================== INDICADORES ====================

def EMA(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calcular_MACD(close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = EMA(close_series, fast)
    ema_slow = EMA(close_series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    macd_slope = macd_line.diff()
    return macd_line, signal_line, histogram, macd_slope, ema_fast, ema_slow

def calcular_RSI_Wilder(close_series: pd.Series, period: int = 14) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==================== FORECASTING (ARIMA) ====================

def forecast_last_step(series: pd.Series, horizons: List[int]) -> Dict[int, float]:
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

def evaluar_indicadores_multi_horizon(ticker: str, close: pd.Series,
                                      forecast_horizons: List[int]) -> Dict:
    macd_line, signal_line, hist, macd_slope, ema_fast, ema_slow = calcular_MACD(close)
    rsi = calcular_RSI_Wilder(close)

    macd_now = float(np.nan_to_num(macd_line.iloc[-1], nan=0.0))
    signal_now = float(np.nan_to_num(signal_line.iloc[-1], nan=0.0))
    macd_slope_now = float(np.nan_to_num(macd_slope.iloc[-1], nan=0.0))
    rsi_now = float(np.nan_to_num(rsi.iloc[-1], nan=50.0))

    macd_fc = forecast_last_step(macd_line, forecast_horizons)
    rsi_fc  = forecast_last_step(rsi,       forecast_horizons)

    results = {
        "ticker": ticker,
        "close": close,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "hist": hist,
        "macd_slope": macd_slope,
        "rsi": rsi,
        "macd_now": macd_now,
        "signal_now": signal_now,
        "macd_slope_now": macd_slope_now,
        "rsi_now": rsi_now,
    }

    for h in forecast_horizons:
        macd_f = float(macd_fc[h])
        rsi_f  = float(rsi_fc[h])

        # Estados
        if macd_now > signal_now and macd_slope_now > 0 and macd_f > macd_now:
            macd_state = "bullish"
        elif macd_now < signal_now and macd_slope_now < 0 and macd_f < macd_now:
            macd_state = "bearish"
        else:
            macd_state = "neutral"

        if rsi_now >= 70 or rsi_f >= 70:
            rsi_state = "overbought"
        elif rsi_now <= 30 or rsi_f <= 30:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

        # Regla de señal
        if macd_state == "bullish" and rsi_state != "overbought":
            trend = "bullish"; recommendation = "BUY"
        elif macd_state == "bearish" and rsi_state != "oversold":
            trend = "bearish"; recommendation = "SELL"
        else:
            trend = "neutral"; recommendation = "HOLD"

        results[f"horizon_{h}_days"] = {
            "macd_forecast": macd_f,
            "rsi_forecast": rsi_f,
            "macd_state": macd_state,
            "rsi_state": rsi_state,
            "trend": trend,
            "recommendation": recommendation,
        }

    return results

# ==================== GRÁFICAS ====================

def plot_price_ema(res: Dict):
    ticker = res["ticker"]; close = res["close"]; ema_fast = res["ema_fast"]; ema_slow = res["ema_slow"]
    fig, ax = plt.subplots(figsize=(10, 5))
    close.plot(ax=ax, label="Close")
    ema_fast.plot(ax=ax, label="EMA 12")
    ema_slow.plot(ax=ax, label="EMA 26")
    ax.set_title(f"{ticker} — Precio y EMAs"); ax.set_xlabel("Fecha"); ax.set_ylabel("Precio")
    ax.legend(); ensure_output_dir()
    fig.savefig(OUTPUT_DIR / f"{ticker}_precio_EMA.png", dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_macd(res: Dict, horizons: List[int]):
    ticker = res["ticker"]; macd = res["macd_line"]; signal = res["signal_line"]; hist = res["hist"]
    fig, ax = plt.subplots(figsize=(10, 4))
    macd.plot(ax=ax, label="MACD"); signal.plot(ax=ax, label="Signal")
    ax.bar(hist.index, hist, label="Histogram")
    txt = " | ".join([f"h{h}d: {res[f'horizon_{h}_days']['macd_forecast']:.4f}" for h in horizons])
    ax.set_title(f"{ticker} — MACD (Forecast → {txt})"); ax.set_xlabel("Fecha"); ax.set_ylabel("Valor")
    ax.legend(); ensure_output_dir()
    fig.savefig(OUTPUT_DIR / f"{ticker}_MACD.png", dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_rsi(res: Dict, horizons: List[int]):
    ticker = res["ticker"]; rsi = res["rsi"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rsi.plot(ax=ax, label="RSI (Wilder)")
    ax.axhline(70, linestyle="--"); ax.axhline(30, linestyle="--")
    txt = " | ".join([f"h{h}d: {res[f'horizon_{h}_days']['rsi_forecast']:.2f}" for h in horizons])
    ax.set_title(f"{ticker} — RSI (70/30) (Forecast → {txt})"); ax.set_xlabel("Fecha"); ax.set_ylabel("RSI")
    ax.legend(); ensure_output_dir()
    fig.savefig(OUTPUT_DIR / f"{ticker}_RSI.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ==================== CSV ====================

def export_signals_to_csv(results: List[Dict], horizons: List[int]):
    rows = []
    for r in results:
        base = {
            "ticker": r["ticker"],
            "macd_now": r["macd_now"],
            "signal_now": r["signal_now"],
            "macd_slope_now": r["macd_slope_now"],
            "rsi_now": r["rsi_now"],
        }
        for h in horizons:
            x = r[f"horizon_{h}_days"]
            rows.append(base | {
                "horizon": h,
                "macd_forecast": x["macd_forecast"],
                "rsi_forecast": x["rsi_forecast"],
                "macd_state": x["macd_state"],
                "rsi_state": x["rsi_state"],
                "trend": x["trend"],
                "recommendation": x["recommendation"],
            })
    df = pd.DataFrame(rows)
    ensure_output_dir()
    path = OUTPUT_DIR / f"signals_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(path, index=False)
    print(f"[CSV] Señales guardadas en: {path}")

# ==================== SIMULADOR DE ÓRDENES (sin broker) ====================

def sim_trade_from_signal(positions: Dict[str, int], symbol: str, signal: str,
                          target_qty: int = 1, long_only: bool = True):
    """
    Simula órdenes para dejar rastro en un CSV (no contacta a ningún broker).
    """
    cur = positions.get(symbol, 0)
    logs = []
    if signal == "BUY":
        tgt = max(target_qty, 0)
        if cur < tgt:
            qty = tgt - cur
            positions[symbol] = tgt
            logs.append({"symbol": symbol, "side": "BUY", "qty": qty})
        else:
            logs.append({"symbol": symbol, "side": "HOLD-BUY", "qty": 0})
    elif signal == "SELL":
        if long_only:
            if cur > 0:
                qty = cur
                positions[symbol] = 0
                logs.append({"symbol": symbol, "side": "SELL-CLOSE", "qty": qty})
            else:
                logs.append({"symbol": symbol, "side": "HOLD-NO-LONG", "qty": 0})
        else:
            tgt = -max(target_qty, 0)
            if cur > tgt:
                qty = cur - tgt
                positions[symbol] = tgt
                logs.append({"symbol": symbol, "side": "SELL", "qty": qty})
            else:
                logs.append({"symbol": symbol, "side": "HOLD-ALREADY-SHORT", "qty": 0})
    else:
        logs.append({"symbol": symbol, "side": "HOLD", "qty": 0})
    return logs

def export_sim_orders(logs: List[Dict]):
    if not logs: 
        return
    df = pd.DataFrame(logs)
    ensure_output_dir()
    path = OUTPUT_DIR / f"sim_orders_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(path, index=False)
    print(f"[SIM] Órdenes simuladas guardadas en: {path}")

# ==================== MAIN ====================

def main():
    closes = descargar_cierres_yf(TICKERS, START_DATE, END_DATE)

    resultados = []
    for sym in TICKERS:
        close = closes[sym]
        if close.empty:
            print(f"[WARN] {sym}: sin datos.")
            continue
        res = evaluar_indicadores_multi_horizon(sym, close, FORECAST_HORIZONS)
        resultados.append(res)

        # Impresión corta
        print(f"Ticker: {sym}")
        print(f"  MACD ahora: {res['macd_now']:.4f}  Señal: {res['signal_now']:.4f}  RSI: {res['rsi_now']:.2f}")
        for h in FORECAST_HORIZONS:
            x = res[f"horizon_{h}_days"]
            print(f"  Horizonte {h}d → MACDf: {x['macd_forecast']:.4f}  RSIf: {x['rsi_forecast']:.2f}  "
                  f"MACD: {x['macd_state']}  RSI: {x['rsi_state']}  {x['trend']} → {x['recommendation']}")
        print("-"*60)

        # Gráficas
        plot_price_ema(res)
        plot_macd(res, FORECAST_HORIZONS)
        plot_rsi(res, FORECAST_HORIZONS)

    # CSV con señales
    if resultados:
        export_signals_to_csv(resultados, FORECAST_HORIZONS)

    # Simulador de órdenes usando el horizonte elegido
    h = TRADE_HORIZON
    positions = {}
    sim_logs = []
    print(f"\n== SIM Trading con horizonte {h}d ==")
    for res in resultados:
        sym = res["ticker"]
        sig = res[f"horizon_{h}_days"]["recommendation"]
        sim_logs += sim_trade_from_signal(positions, sym, sig, target_qty=TARGET_QTY, long_only=LONG_ONLY)
        print(f"[{sym}] → {sig}, pos:{positions.get(sym,0)}")

    export_sim_orders(sim_logs)
    print("\nListo. Revisa carpeta 'outputs/' (PNGs, CSV de señales y sim_orders).")

if __name__ == "__main__":
    main()
