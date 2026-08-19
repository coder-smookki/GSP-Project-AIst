"""Точка входа бэкенда интерфейса.

Запуск::

    uvicorn custom_gui.server.main:app --host 0.0.0.0 --port 8000
"""

import logging

from custom_gui.server.app import create_app

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

app = create_app()
