FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY setup.py .

COPY models models

RUN python -c "from app.ml_logic.data import get_financial_data; get_financial_data(tickers=['BTC-USD','ETH-USD','SPY'], period_years=10)"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.api.fast:app --host 0.0.0.0 --port ${PORT}"]
