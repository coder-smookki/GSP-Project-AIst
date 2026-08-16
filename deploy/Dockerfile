FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir "aegra-api>=0.9.0,<0.10.0"

COPY pyproject.toml .
COPY . .
RUN pip install --no-cache-dir .

COPY deploy/aegra.json ./aegra.json

# Переоценка вынесена из образа: в dev её считает хост-сервис
# (deploy/rerank_server.py, MPS на macOS), в prod — отдельный GPU-контейнер
# (docker-compose.reranker.yml, CUDA). Здесь контейнер только ходит к ним по
# HTTP через RERANK_ENDPOINT; torch/sentence-transformers и веса модели в
# образ не нужны (все импорты в rerank.py ленивые). Если сервис переоценки
# недоступен, переоценка бесшумно отключается до перезапуска.
ENV PYTHONPATH=/app
EXPOSE 2026

CMD ["uvicorn", "aegra_api.main:app", "--host", "0.0.0.0", "--port", "2026"]