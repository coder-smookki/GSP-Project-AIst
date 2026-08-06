FROM python:3.12-slim-bookworm

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir "aegra-api>=0.9.0,<0.10.0"

COPY . .
RUN pip install --no-cache-dir .
COPY deploy/aegra.json ./aegra.json

ENV PYTHONPATH=/app
EXPOSE 2026

CMD ["uvicorn", "aegra_api.main:app", "--host", "0.0.0.0", "--port", "2026"]
