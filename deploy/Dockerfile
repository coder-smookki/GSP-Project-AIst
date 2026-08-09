FROM python:3.12-slim-bookworm

WORKDIR /app

# torch ставим первым: CPU-колесо намного меньше CUDA-сборки. Индекс
# download.pytorch.org/whl/cpu экономит порядка 3 ГБ на Linux.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "aegra-api>=0.9.0,<0.10.0"

COPY pyproject.toml .
COPY . .
RUN pip install --no-cache-dir ".[rerank]"
COPY deploy/aegra.json ./aegra.json

# Переоценка кросс-энкодером: веса запекаются в образ при сборке, чтобы
# RERANK_ENABLED=1 работал без обращений наружу (требование кейса — ноль
# внешних API). snapshot_download кладёт файлы модели в HF_HOME (тот же кэш,
# что transformers читает офлайн) и не грузит веса в память — память нужна
# только на диске. Сверяйте с RERANK_MODEL из .env — другую модель нужно
# запекать отдельной пересборкой.
ENV HF_HOME=/opt/rerank-cache
RUN HF_HUB_DISABLE_XET=1 python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')" \
 && echo "reranker baked at build" >> /opt/reranker-baked
ENV PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
EXPOSE 2026

CMD ["uvicorn", "aegra_api.main:app", "--host", "0.0.0.0", "--port", "2026"]
