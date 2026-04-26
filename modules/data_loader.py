"""
Data Loader Module
Fetches OHLCV data from Yahoo Finance for financial underlyings.
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional

# Default ticker mapping
DEFAULT_TICKERS: Dict[str, str] = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "VVIX": "^VVIX",
}


def load_ticker(
    symbol: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker (e.g. "^GSPC", "^VIX")
    start : str
        Start date in YYYY-MM-DD format
    end : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    interval : str
        Data frequency: "1d", "1wk", "1mo"

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex
    """
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    # Standardise column names
    df.columns = [c.capitalize() for c in df.columns]
    return df


def load_universe(
    tickers: Optional[Dict[str, str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple tickers.

    Parameters
    ----------
    tickers : dict, optional
        Mapping of name -> Yahoo symbol. Defaults to DEFAULT_TICKERS.
    start, end, interval : see load_ticker

    Returns
    -------
    dict
        {name: DataFrame}
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    universe: Dict[str, pd.DataFrame] = {}
    for name, symbol in tickers.items():
        print(f"  Downloading {name} ({symbol}) ...", end=" ")
        try:
            df = load_ticker(symbol, start=start, end=end, interval=interval)
            universe[name] = df
            print(f"OK  ({len(df)} rows)")
        except Exception as exc:
            print(f"FAILED — {exc}")

    return universe


def align_universe(
    universe: Dict[str, pd.DataFrame],
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Align multiple tickers on a common date index.
    Returns a single DataFrame with columns = ticker names.

    Parameters
    ----------
    universe : dict
        Output of load_universe
    price_col : str
        Which OHLCV column to keep for each ticker

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by Date
    """
    series = {
        name: df[price_col].rename(name)
        for name, df in universe.items()
        if price_col in df.columns
    }
    aligned = pd.concat(series, axis=1)
    aligned = aligned.dropna()
    return aligned


def get_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute simple returns over `periods` trading days."""
    return prices.pct_change(periods=periods)
