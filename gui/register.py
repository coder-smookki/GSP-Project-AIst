"""Страница самостоятельной регистрации.

Chainlit собран на FastAPI, поэтому своя страница добавляется маршрутом в то
же приложение - отдельного сервиса и отдельного порта не появляется.

**Регистрация закрыта кодом приглашения.** Заказчик прямо сказал, что учётные
записи заводит администратор, и открытая регистрация этому противоречила бы:
любой, кто дотянулся до порта, завёл бы себе доступ к нормативной базе.
Компромисс - администратор выдаёт код (``REGISTRATION_CODE``), человек
регистрируется сам. Кода нет - страница недоступна вовсе, а не пускает всех.
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import time

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from gui.db import create_user, get_user

logger = logging.getLogger(__name__)

#: Допустимое имя пользователя. Оно попадает в базу и в заголовки, поэтому
#: набор символов узкий.
_USERNAME = re.compile(r"^[A-Za-z0-9._-]{3,64}$")

#: Минимальная длина пароля. Ниже - подбор становится делом минут.
_MIN_PASSWORD = 8

#: Сколько попыток регистрации разрешено с одного адреса за окно.
_MAX_ATTEMPTS = 10

#: Окно ограничения, с.
_WINDOW = 300.0

#: Попытки по адресу: время последних обращений.
_attempts: dict[str, list[float]] = {}


def _code() -> str:
    """Возвращает код приглашения из окружения.

    Returns:
        str: Код; пустая строка означает, что регистрация выключена.
    """
    return (os.getenv("REGISTRATION_CODE") or "").strip()


def _too_many(request: Request) -> bool:
    """Проверяет и учитывает частоту обращений с адреса.

    Args:
        request: Входящий запрос.

    Returns:
        bool: ``True``, если предел попыток исчерпан.
    """
    who = request.client.host if request.client else "unknown"
    now = time.monotonic()
    seen = [t for t in _attempts.get(who, []) if now - t < _WINDOW]
    seen.append(now)
    # Словарь чистится по ходу дела: иначе он растёт от каждого нового
    # адреса и живёт до перезапуска.
    _attempts[who] = seen
    for key in [k for k, v in _attempts.items() if not v or now - v[-1] > _WINDOW]:
        _attempts.pop(key, None)
    return len(seen) > _MAX_ATTEMPTS


def _page(message: str = "", ok: bool = False) -> str:
    """Собирает страницу регистрации.

    Стиль фирменный и встроенный: внешних адресов быть не должно, замкнутый
    контур - блокирующее требование.

    Args:
        message: Сообщение под формой.
        ok: Сообщение об успехе, а не об ошибке.

    Returns:
        str: HTML страницы.
    """
    tone = "#3a7f4c" if ok else "#b85c1c"
    note = f'<p class="msg" style="color:{tone}">{message}</p>' if message else ""
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>АИст — регистрация</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
    background:#081830; color:#e9ecef;
    font:15px/1.5 -apple-system,'Segoe UI',system-ui,Arial,sans-serif; }}
  .card {{ width:min(400px,92vw); padding:30px; background:#101f39;
    border:1px solid #18365e; border-radius:14px; }}
  h1 {{ margin:0 0 4px; font-size:21px; letter-spacing:-.02em; }}
  .sub {{ margin:0 0 20px; color:#7ba2e8; font-size:13px; }}
  label {{ display:block; margin:14px 0 5px; font-size:12.5px; color:#a7aeb6; }}
  input {{ width:100%; padding:10px 12px; font:inherit; color:#e9ecef;
    background:#0c1728; border:1px solid #22406e; border-radius:9px; }}
  input:focus {{ outline:2px solid #4aafe4; outline-offset:1px; }}
  button {{ width:100%; margin-top:20px; padding:11px; font:inherit;
    font-weight:600; color:#fff; background:#086eb6; border:0;
    border-radius:9px; cursor:pointer; }}
  button:hover {{ background:#18365e; }}
  .msg {{ margin:14px 0 0; font-size:13px; }}
  .foot {{ margin-top:18px; font-size:12.5px; color:#7b838c; }}
  a {{ color:#4aafe4; }}
</style></head>
<body>
  <form class="card" method="post" action="/register">
    <h1>Регистрация</h1>
    <p class="sub">Код приглашения выдаёт администратор системы</p>
    <label for="u">Логин</label>
    <input id="u" name="username" autocomplete="username" required>
    <label for="p">Пароль</label>
    <input id="p" name="password" type="password"
           autocomplete="new-password" minlength="{_MIN_PASSWORD}" required>
    <label for="c">Код приглашения</label>
    <input id="c" name="code" autocomplete="off" required>
    <button type="submit">Создать учётную запись</button>
    {note}
    <p class="foot">Уже есть доступ? <a href="/login">Войти</a></p>
  </form>
</body></html>"""


def register_routes(app) -> None:
    """Подключает страницу регистрации к приложению Chainlit.

    Args:
        app: Приложение FastAPI, на котором работает Chainlit.
    """

    @app.get("/register", response_class=HTMLResponse)
    async def registration_page() -> HTMLResponse:
        """Отдаёт форму регистрации.

        Returns:
            HTMLResponse: Страница или 404, если регистрация выключена.
        """
        if not _code():
            # Именно 404, а не «регистрация выключена»: посторонний не должен
            # узнавать, что такая страница вообще предусмотрена.
            return HTMLResponse("Not Found", status_code=404)
        return HTMLResponse(_page())

    # response_model=None: возвращается либо страница, либо переход, а по
    # объединению типов FastAPI модель ответа построить не может.
    @app.post("/register", response_class=HTMLResponse, response_model=None)
    async def registration_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        code: str = Form(...),
    ) -> HTMLResponse | RedirectResponse:
        """Заводит учётную запись и уводит на вход.

        Args:
            request: Входящий запрос.
            username: Желаемый логин.
            password: Пароль.
            code: Код приглашения.

        Returns:
            HTMLResponse | RedirectResponse: Страница с ошибкой либо переход
            на страницу входа.
        """
        expected = _code()
        if not expected:
            return HTMLResponse("Not Found", status_code=404)

        if _too_many(request):
            logger.warning("регистрация: превышена частота обращений")
            return HTMLResponse(
                _page("Слишком много попыток. Повторите через несколько минут."),
                status_code=429,
            )

        # Сравнение постоянного времени по байтам: на строках оно не
        # принимает не-ASCII и уронило бы страницу на кириллическом коде.
        if not hmac.compare_digest(
            code.strip().encode("utf-8"), expected.encode("utf-8")
        ):
            logger.warning("регистрация: неверный код приглашения")
            return HTMLResponse(_page("Неверный код приглашения."), status_code=403)

        name = username.strip()
        if not _USERNAME.match(name):
            return HTMLResponse(
                _page(
                    "Логин: латиница, цифры, точка, дефис или подчёркивание, "
                    "от 3 до 64 знаков."
                ),
                status_code=400,
            )
        if len(password) < _MIN_PASSWORD:
            return HTMLResponse(
                _page(f"Пароль короче {_MIN_PASSWORD} знаков."), status_code=400
            )

        if await get_user(name) is not None:
            return HTMLResponse(_page("Такой логин уже занят."), status_code=409)

        if not await create_user(name, password):
            return HTMLResponse(
                _page("Не удалось завести учётную запись."), status_code=409
            )

        logger.info("зарегистрирован пользователь %s", name)
        return RedirectResponse("/login", status_code=303)
