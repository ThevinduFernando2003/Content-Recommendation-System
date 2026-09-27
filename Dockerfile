FROM python:3.11-slim AS builder

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY src ./src
COPY scripts/train.py ./scripts/train.py
COPY data/raw/netflix_titles.csv ./data/raw/netflix_titles.csv
RUN python scripts/train.py

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/config.yaml ./config.yaml
COPY --from=builder /app/models ./models
COPY app ./app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PORT=8000 \
    API_BASE_URL=http://127.0.0.1:8000

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health')"
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
