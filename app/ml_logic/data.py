import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path(__file__).resolve().parents[2] / 'data_folder' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = ['BTC-USD']
DEFAULT_YEARS = 10


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('/', '_')}.csv"


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Aucune donnée trouvée pour {ticker}")
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index.name = 'Date'
    return df


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return df


def _save_cache(ticker: str, df: pd.DataFrame):
    df.to_csv(_cache_path(ticker))


def get_financial_data(
    tickers: list[str] = DEFAULT_TICKERS,
    period_years: int = DEFAULT_YEARS,
    force_refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Télécharge ou met à jour les données OHLCV pour une liste de tickers.

    Args:
        tickers: liste de tickers yfinance (ex: ['BTC-USD', 'ETH-USD', 'SPY'])
        period_years: nombre d'années d'historique si pas de cache
        force_refresh: ignore le cache et retélécharge tout

    Returns:
        dict { ticker: DataFrame } avec colonnes Open, High, Low, Close, Volume
    """
    end = datetime.today().strftime('%Y-%m-%d')
    start_full = (datetime.today() - timedelta(days=365 * period_years)).strftime('%Y-%m-%d')

    results = {}

    for ticker in tickers:
        print(f"[{ticker}] ", end='')

        cached = _load_cache(ticker) if not force_refresh else None

        if cached is None:
            print(f"téléchargement complet ({period_years} ans)...")
            df = _download(ticker, start=start_full, end=end)

        else:
            last_date = cached.index[-1]
            days_missing = (datetime.today() - last_date).days

            if days_missing <= 1:
                print(f"cache à jour ({len(cached)} lignes)")
                results[ticker] = cached
                continue

            print(f"mise à jour depuis {last_date.date()} ({days_missing} jours manquants)...")
            new = _download(ticker, start=last_date.strftime('%Y-%m-%d'), end=end)
            df = pd.concat([cached, new]).drop_duplicates()

        _save_cache(ticker, df)
        print(f"  → {len(df)} lignes sauvegardées")
        results[ticker] = df

    return results
