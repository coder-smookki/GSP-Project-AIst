"""Хост-сервис переоценки кандидатов (MPS на macOS).

Colima не пробрасывает GPU, и инференс кросс-энкодера в контейнере в ~16 раз
медленнее нативного macOS-торча. Этот сервис запускается на хосте рядом с
проектным ``.venv`` и считает оценки на Metal (MPS); контейнер ходит к нему
через ``host.docker.internal`` (см. ``RERANK_ENDPOINT`` в настройках).

Запуск::

    .venv/bin/python deploy/rerank_server.py

Окружение: ``RERANK_MODEL`` (по умолчанию bge-reranker-v2-m3),
``RERANK_PORT`` (по умолчанию 8899), ``RERANK_MAX_LENGTH`` (по умолчанию 512).

Сервис слушает 0.0.0.0, потому что контейнер colima обращается к хосту через
шлюз VM, а не через loopback. Модель грузится один раз при старте.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rerank-server")

MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
PORT = int(os.environ.get("RERANK_PORT", "8899"))
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))

_lock = Lock()
_model: Any = None
_device: str | None = None


def _load_model() -> None:
    global _model, _device
    import torch
    from sentence_transformers import CrossEncoder

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # fp16 ускоряет только cuda/mps; на CPU оно лишь деградирует (~2.4x медленнее).
    dtype = torch.float16 if device != "cpu" else torch.float32
    _model = CrossEncoder(
        MODEL,
        max_length=MAX_LENGTH,
        device=device,
        model_kwargs={"torch_dtype": dtype},
    )
    _device = device
    logger.info("модель %s загружена на %s", MODEL, device)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"ok": True, "model": MODEL, "device": _device})

    def do_POST(self) -> None:
        if self.path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(request["query"])
            candidates = [str(item) for item in request["candidates"]]
        except (ValueError, KeyError, TypeError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return
        try:
            with _lock:
                scores = _model.predict([(query, text) for text in candidates])
            self._send(200, {"scores": [float(score) for score in scores]})
        except Exception as exc:
            logger.exception("ошибка инференса")
            self._send(500, {"error": str(exc)})


def main() -> None:
    _load_model()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    logger.info("rerank-сервер слушает 0.0.0.0:%d", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
