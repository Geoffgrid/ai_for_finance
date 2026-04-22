import shutil
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api import fast


client = TestClient(fast.app)


def test_predict_fetches_and_appends_missing_market_rows(monkeypatch):
    cache_csv = Path("data_folder/cache/BTC-USD.csv")
    backup_csv = cache_csv.with_suffix(".csv.test_backup")

    if backup_csv.exists():
        backup_csv.unlink()
    shutil.copy2(cache_csv, backup_csv)

    original_df = pd.read_csv(cache_csv)
    original_rows = len(original_df)
    truncated_df = original_df.iloc[:-3].copy()
    truncated_df.to_csv(cache_csv, index=False)

    # Keep prediction deterministic while exercising real data refresh path.
    monkeypatch.setattr(
        fast,
        "my_prediction_function",
        lambda open_price, high, low, close, volume: [1],
    )

    try:
        response = client.get(
            "/predict",
            params={
                "open_price": 10.0,
                "high": 12.0,
                "low": 9.5,
                "close": 11.0,
                "volume": 1000,
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": 1}

        refreshed_df = pd.read_csv(cache_csv)
        assert len(refreshed_df) >= original_rows
        assert refreshed_df["Date"].duplicated().sum() == 0
    finally:
        shutil.move(str(backup_csv), str(cache_csv))
