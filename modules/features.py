"""
Feature Engineering Module
Builds technical indicators on OHLCV price series.

Supported features
------------------
Trend          : SMA, EMA (multiple windows), MA slopes, MA distances
Momentum       : RSI, MACD, Rate-of-Change, Stochastic %K/%D, Williams %R, CCI
Volatility     : Bollinger Bands (%B, width), ATR, Historical Volatility
Volume         : OBV, Volume z-score, Volume ratio
Cross-asset    : VIX level/change/z-score, VVIX/VIX ratio, VIX regime
Target         : forward_return, forward_label (binary)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _slope(series: pd.Series, window: int) -> pd.Series:
    """Linear regression slope over a rolling window (normalised by std)."""
    def _reg_slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    return series.rolling(window).apply(_reg_slope, raw=True)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / (sigma + 1e-9)


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame, col: str = "Close",
                        windows=(5, 10, 20, 50, 200),
                        ema_windows=(12, 26)) -> pd.DataFrame:
    """SMA and EMA columns."""
    for w in windows:
        df[f"SMA_{w}"] = df[col].rolling(w).mean()
    for w in ema_windows:
        df[f"EMA_{w}"] = df[col].ewm(span=w, adjust=False).mean()
    return df


def add_ma_slopes(df: pd.DataFrame, windows=(5, 20, 50),
                  slope_window: int = 5) -> pd.DataFrame:
    """Slope (linear regression) of each moving average over `slope_window` bars."""
    for w in windows:
        col = f"SMA_{w}"
        if col in df.columns:
            df[f"slope_SMA_{w}"] = _slope(df[col], slope_window)
    return df


def add_ma_distances(df: pd.DataFrame, col: str = "Close",
                     windows=(20, 50, 200)) -> pd.DataFrame:
    """% distance from price to each moving average."""
    for w in windows:
        sma_col = f"SMA_{w}"
        if sma_col in df.columns:
            df[f"dist_SMA_{w}_pct"] = (df[col] - df[sma_col]) / df[sma_col]
    return df


def add_rsi(df: pd.DataFrame, col: str = "Close",
            window: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df[f"RSI_{window}"] = 100 - 100 / (1 + rs)
    return df


def add_bollinger_bands(df: pd.DataFrame, col: str = "Close",
                        window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, lower, %B, width."""
    mid = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    df["BB_upper"] = upper
    df["BB_lower"] = lower
    df["BB_mid"] = mid
    df["BB_width"] = (upper - lower) / (mid + 1e-9)
    df["BB_pct_b"] = (df[col] - lower) / (upper - lower + 1e-9)
    return df


def add_macd(df: pd.DataFrame, col: str = "Close",
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = macd_line - signal_line
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range — requires High, Low, Close columns."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return df
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df[f"ATR_{window}"] = tr.ewm(alpha=1 / window, adjust=False).mean()
    # Normalise ATR by price for comparability
    df[f"ATR_{window}_pct"] = df[f"ATR_{window}"] / df["Close"]
    return df


def add_rate_of_change(df: pd.DataFrame, col: str = "Close",
                       windows=(5, 10, 21, 63)) -> pd.DataFrame:
    """Rate of Change (momentum): (price_t / price_{t-n}) - 1."""
    for w in windows:
        df[f"ROC_{w}"] = df[col].pct_change(periods=w)
    return df


def add_stochastic(df: pd.DataFrame, k_window: int = 14,
                   d_window: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator %K and %D."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return df
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(d_window).mean()
    return df


def add_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Williams %R."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return df
    high_max = df["High"].rolling(window).max()
    low_min = df["Low"].rolling(window).min()
    df["Williams_R"] = -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-9)
    return df


def add_cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return df
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["CCI"] = (tp - ma_tp) / (0.015 * mad + 1e-9)
    return df


def add_historical_volatility(df: pd.DataFrame, col: str = "Close",
                               windows=(21, 63)) -> pd.DataFrame:
    """Annualised historical volatility (std of log returns)."""
    log_ret = np.log(df[col] / df[col].shift(1))
    for w in windows:
        df[f"HVol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
    return df


def add_volume_features(df: pd.DataFrame,
                        volume_col: str = "Volume") -> pd.DataFrame:
    """OBV, volume z-score, volume ratio (vs 20d avg)."""
    if volume_col not in df.columns:
        return df
    # OBV
    sign = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (sign * df[volume_col]).cumsum()
    # Volume z-score (21d)
    df["Vol_zscore"] = _zscore(df[volume_col], 21)
    # Volume ratio vs 20-day average
    df["Vol_ratio"] = df[volume_col] / (df[volume_col].rolling(20).mean() + 1e-9)
    return df


def add_vix_features(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Add VIX-derived features to a base DataFrame.

    Parameters
    ----------
    df   : base DataFrame (SPX)
    vix  : VIX Close price aligned to df.index
    """
    vix = vix.reindex(df.index)
    df["VIX_level"] = vix
    df["VIX_change_1d"] = vix.pct_change(1)
    df["VIX_change_5d"] = vix.pct_change(5)
    df["VIX_zscore_21"] = _zscore(vix, 21)
    df["VIX_zscore_63"] = _zscore(vix, 63)
    # VIX regime: 0=low(<15), 1=normal(15-25), 2=high(>25)
    df["VIX_regime"] = pd.cut(
        vix, bins=[0, 15, 25, np.inf], labels=[0, 1, 2]
    ).astype(float)
    return df


def add_vvix_features(df: pd.DataFrame, vvix: pd.Series,
                      vix: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Add VVIX-derived features.

    Parameters
    ----------
    df   : base DataFrame
    vvix : VVIX Close series aligned to df.index
    vix  : VIX Close series (optional, for VVIX/VIX ratio)
    """
    vvix = vvix.reindex(df.index)
    df["VVIX_level"] = vvix
    df["VVIX_change_1d"] = vvix.pct_change(1)
    df["VVIX_zscore_21"] = _zscore(vvix, 21)
    if vix is not None:
        vix = vix.reindex(df.index)
        df["VVIX_VIX_ratio"] = vvix / (vix + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def add_forward_return(df: pd.DataFrame, col: str = "Close",
                       horizon: int = 5) -> pd.DataFrame:
    """
    Add forward return and binary label.

    forward_return_{n}  : (Close_{t+n} / Close_t) - 1
    forward_label_{n}   : 1 if forward_return > 0 else 0
    """
    fwd_ret = df[col].pct_change(periods=horizon).shift(-horizon)
    df[f"forward_return_{horizon}"] = fwd_ret
    df[f"forward_label_{horizon}"] = (fwd_ret > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Master pipeline transformer
# ---------------------------------------------------------------------------

class TechnicalFeatureTransformer:
    """
    sklearn-compatible transformer that builds all technical features.

    Designed to work inside a Pipeline. Input must be a DataFrame with
    OHLCV columns and optionally 'VIX' / 'VVIX' columns (from merge).

    Parameters
    ----------
    horizon : int
        Forward return horizon for the target variable (default 5 days)
    add_target : bool
        Whether to append forward_return and forward_label columns
    """

    def __init__(self, horizon: int = 5, add_target: bool = False):
        self.horizon = horizon
        self.add_target = add_target

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        # --- Price-based features (SPX) ---
        df = add_moving_averages(df)
        df = add_ma_slopes(df)
        df = add_ma_distances(df)
        df = add_rsi(df)
        df = add_bollinger_bands(df)
        df = add_macd(df)
        df = add_atr(df)
        df = add_rate_of_change(df)
        df = add_stochastic(df)
        df = add_williams_r(df)
        df = add_cci(df)
        df = add_historical_volatility(df)
        df = add_volume_features(df)

        # --- Cross-asset features ---
        if "VIX_raw" in df.columns:
            df = add_vix_features(df, df["VIX_raw"])
        if "VVIX_raw" in df.columns:
            vix_series = df["VIX_raw"] if "VIX_raw" in df.columns else None
            df = add_vvix_features(df, df["VVIX_raw"], vix=vix_series)

        # --- Target ---
        if self.add_target:
            df = add_forward_return(df, horizon=self.horizon)

        return df


def build_feature_matrix(
    spx: pd.DataFrame,
    vix: Optional[pd.DataFrame] = None,
    vvix: Optional[pd.DataFrame] = None,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Convenience function: merge SPX with VIX/VVIX then compute all features.

    Parameters
    ----------
    spx, vix, vvix : DataFrames from data_loader
    horizon : forward return horizon

    Returns
    -------
    pd.DataFrame
        Full feature matrix including target columns
    """
    df = spx.copy()

    if vix is not None:
        df["VIX_raw"] = vix["Close"].reindex(df.index)
    if vvix is not None:
        df["VVIX_raw"] = vvix["Close"].reindex(df.index)

    transformer = TechnicalFeatureTransformer(horizon=horizon, add_target=True)
    df = transformer.transform(df)

    return df


def get_feature_columns(df: pd.DataFrame, horizon: int = 5) -> list:
    """Return list of feature columns (excluding OHLCV raw and target cols)."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "VIX_raw", "VVIX_raw",
        f"forward_return_{horizon}", f"forward_label_{horizon}",
    }
    return [c for c in df.columns if c not in exclude]
