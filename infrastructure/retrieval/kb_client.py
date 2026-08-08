"""Асинхронный клиент к API базы знаний заказчика (HiHub).

Контракт снят с Postman-коллекции организаторов и проверен живым запросом
08.08.2026, менять его нельзя:

===========================================  ==========================================
Метод                                        Назначение
===========================================  ==========================================
``POST {host}api/auth/logininng``            вход, отдаёт access/refresh токены
``POST {host}api/auth/refresh``              обновление по refresh-токену
``POST {host}api/auth/logout``               отзыв refresh-токена
``GET  {host}api/knowledgebase/section/{id}/articles``  список статей раздела
``GET  {host}api/knowledgebase/article/{id}``           статья целиком
===========================================  ==========================================

Опечатка в ``logininng`` — на стороне источника, воспроизводим как есть.

Особенности, влияющие на реализацию:

* ``access_token`` живёт **3599 секунд**. Индексация всего корпуса дольше
  часа вполне реальна, поэтому обновление токена обязательно, а не опционально.
* Список статей отдаётся постранично в формате Laravel: ``data`` + ``links``
  + ``meta.last_page``.
* Методы создания, изменения и удаления статей закрыты ролевой моделью и
  здесь намеренно не реализованы: клиент доступа только на чтение.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import httpx

logger = logging.getLogger(__name__)

#: Коды, при которых имеет смысл повторить запрос.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Предел размера ответа. Одна статья корпуса весит до 1 МБ; всё, что сильно
#: больше, — аномалия, которую нельзя молча тянуть в память индексатора.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class KnowledgeBaseError(RuntimeError):
    """Базовая ошибка работы с API базы знаний."""


class AuthError(KnowledgeBaseError):
    """Не удалось получить или обновить токен доступа."""


class TransportError(KnowledgeBaseError):
    """Запрос не удался после всех повторов."""


@dataclass(slots=True)
class KnowledgeBaseConfig:
    """Параметры подключения к API базы знаний.

    Args:
        base_url: Корень API, например ``https://hihub.ru/``. Завершающий
            слэш добавляется автоматически.
        email: Логин учётной записи.
        password: Пароль учётной записи. В логи не попадает никогда.
        section_id: Идентификатор раздела с материалами кейса.
        token_name: Произвольная метка токена, требуется API при входе.
        timeout_connect: Таймаут установления соединения, с.
        timeout_read: Таймаут чтения ответа, с. Крупные статьи отдаются
            небыстро, поэтому значение заметно больше таймаута соединения.
        max_retries: Число повторов при временных ошибках.
        backoff_base: Базовая задержка экспоненциального повтора, с.
        backoff_max: Верхняя граница задержки повтора, с.
        per_page: Размер страницы при перечислении статей.
        verify_tls: Проверять ли сертификат сервера. Отключать только для
            стенда с самоподписанным сертификатом.
    """

    base_url: str
    email: str
    password: str = field(repr=False)
    section_id: int
    token_name: str = "aist"
    timeout_connect: float = 10.0
    timeout_read: float = 60.0
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 8.0
    per_page: int = 100
    verify_tls: bool = True

    def __post_init__(self) -> None:
        """Проверяет обязательные поля и нормализует адрес.

        Raises:
            ValueError: Если адрес пуст, не является http(s) или не заданы
                учётные данные.
        """
        self.base_url = self.base_url.strip()
        if not self.base_url:
            raise ValueError("не задан адрес API базы знаний")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"адрес API должен начинаться с http:// или https://, получено "
                f"{self.base_url!r}"
            )
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        if not self.email or not self.password:
            raise ValueError("не заданы учётные данные для API базы знаний")


class KnowledgeBaseClient:
    """Клиент чтения базы знаний с автоматическим обновлением токена.

    Использование:

        async with KnowledgeBaseClient(config) as client:
            async for article in client.iter_section_articles():
                ...

    Клиент потокобезопасен в пределах одного цикла событий: одновременные
    запросы, получившие 401, обновляют токен ровно один раз — остальные
    ждут на блокировке и переиспользуют результат.
    """

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Создаёт клиент.

        Args:
            config: Параметры подключения.
            transport: Транспорт httpx. Подменяется в тестах на
                `httpx.MockTransport`, в проде остаётся ``None``.
        """
        self._config = config
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._auth_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(
                connect=config.timeout_connect,
                read=config.timeout_read,
                write=config.timeout_read,
                pool=config.timeout_connect,
            ),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            follow_redirects=False,
            verify=config.verify_tls,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        """Открывает сессию и выполняет вход.

        Returns:
            Self: Готовый к работе клиент.
        """
        await self.login()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Закрывает сессию, стараясь отозвать refresh-токен.

        Args:
            exc_type: Тип исключения, если блок завершился ошибкой.
            exc: Экземпляр исключения.
            tb: Трассировка.
        """
        try:
            await self.logout()
        except KnowledgeBaseError as logout_error:
            logger.warning("не удалось отозвать токен при выходе: %s", logout_error)
        finally:
            await self._client.aclose()

    # ------------------------------------------------------------------ auth

    async def login(self) -> None:
        """Выполняет вход и запоминает пару токенов.

        Raises:
            AuthError: Если сервер отверг учётные данные или вернул ответ
                без токена доступа.
        """
        payload = {
            "email": self._config.email,
            "password": self._config.password,
            "token_name": self._config.token_name,
        }
        logger.info("вход в базу знаний под учётной записью %s", self._config.email)
        response = await self._request(
            "POST", "api/auth/logininng", json=payload, authenticated=False
        )
        data = self._as_json(response)
        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            raise AuthError("ответ входа не содержит access_token")
        self._access_token = token
        refresh = data.get("refresh_token")
        self._refresh_token = refresh if isinstance(refresh, str) else None
        logger.info(
            "вход выполнен, срок действия токена %s с, обновление %s",
            data.get("expires_in", "неизвестен"),
            "доступно" if self._refresh_token else "недоступно",
        )

    async def _reauthenticate(self) -> None:
        """Обновляет токен доступа, при неудаче выполняет полный вход.

        Вызывается под блокировкой: одновременные 401 не должны порождать
        стадо запросов обновления.

        Raises:
            AuthError: Если и обновление, и повторный вход не удались.
        """
        if self._refresh_token:
            try:
                response = await self._request(
                    "POST",
                    "api/auth/refresh",
                    json={
                        "refresh_token": self._refresh_token,
                        "token_name": self._config.token_name,
                    },
                    authenticated=False,
                )
                data = self._as_json(response)
                token = data.get("access_token")
                if isinstance(token, str) and token:
                    self._access_token = token
                    new_refresh = data.get("refresh_token")
                    if isinstance(new_refresh, str) and new_refresh:
                        self._refresh_token = new_refresh
                    logger.info("токен доступа обновлён по refresh-токену")
                    return
                logger.warning("обновление вернуло ответ без access_token")
            except KnowledgeBaseError as refresh_error:
                logger.warning("обновление токена не удалось: %s", refresh_error)
        logger.info("выполняется повторный вход")
        await self.login()

    async def logout(self) -> None:
        """Отзывает refresh-токен на сервере.

        Ошибки выхода не считаются фатальными: токен всё равно истечёт сам.
        """
        if not self._refresh_token:
            return
        try:
            await self._request(
                "POST",
                "api/auth/logout",
                json={"refresh_token": self._refresh_token},
            )
            logger.info("сессия базы знаний закрыта")
        finally:
            self._access_token = None
            self._refresh_token = None

    # --------------------------------------------------------------- запросы

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> httpx.Response:
        """Выполняет запрос с повторами и однократным обновлением токена.

        Args:
            method: HTTP-метод.
            url: Путь относительно ``base_url``.
            json: Тело запроса.
            params: Параметры строки запроса.
            authenticated: Добавлять ли заголовок ``Authorization``.

        Returns:
            httpx.Response: Успешный ответ (2xx).

        Raises:
            AuthError: Если сервер отвечает 401 и после обновления токена.
            TransportError: Если запрос не удался после всех повторов либо
                сервер вернул неустранимую ошибку.
        """
        attempts = self._config.max_retries + 1
        last_error: str = ""
        refreshed = False

        for attempt in range(1, attempts + 1):
            headers: dict[str, str] = {}
            if authenticated and self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"

            try:
                response = await self._client.request(
                    method, url, json=json, params=params, headers=headers
                )
            except httpx.TimeoutException as exc:
                last_error = f"таймаут: {exc}"
            except httpx.TransportError as exc:
                last_error = f"сетевая ошибка: {exc}"
            else:
                if response.status_code == 401 and authenticated and not refreshed:
                    refreshed = True
                    logger.info("получен 401 на %s, обновляем токен", url)
                    async with self._auth_lock:
                        await self._reauthenticate()
                    continue
                if response.status_code < 400:
                    self._guard_size(response, url)
                    return response
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}"
                else:
                    raise TransportError(
                        f"{method} {url}: HTTP {response.status_code} — "
                        f"{self._safe_excerpt(response)}"
                    )

            if attempt < attempts:
                delay = self._backoff_delay(attempt, locals().get("response"))
                logger.warning(
                    "%s %s: попытка %d из %d не удалась (%s), повтор через %.1f с",
                    method,
                    url,
                    attempt,
                    attempts,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)

        raise TransportError(f"{method} {url}: исчерпаны повторы — {last_error}")

    def _backoff_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Считает задержку перед повтором.

        Учитывает заголовок ``Retry-After``, если сервер его прислал, иначе
        применяет экспоненциальный рост со случайным разбросом: без разброса
        параллельные индексаторы синхронно бьют по одному и тому же моменту.

        Args:
            attempt: Номер завершившейся попытки, начиная с 1.
            response: Последний ответ, если он был получен.

        Returns:
            float: Задержка в секундах, не больше `backoff_max`.
        """
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    return min(float(header), self._config.backoff_max)
                except ValueError:
                    logger.debug("не удалось разобрать Retry-After: %r", header)
        window = min(
            self._config.backoff_base * (2 ** (attempt - 1)), self._config.backoff_max
        )
        return window * (0.5 + random.random() / 2)

    @staticmethod
    def _guard_size(response: httpx.Response, url: str) -> None:
        """Проверяет, что ответ не превышает допустимый размер.

        Args:
            response: Ответ сервера.
            url: Путь запроса, попадает в текст ошибки.

        Raises:
            TransportError: Если тело ответа больше `MAX_RESPONSE_BYTES`.
        """
        size = len(response.content)
        if size > MAX_RESPONSE_BYTES:
            raise TransportError(
                f"{url}: ответ {size} байт превышает предел {MAX_RESPONSE_BYTES}"
            )

    @staticmethod
    def _safe_excerpt(response: httpx.Response, limit: int = 200) -> str:
        """Возвращает безопасный для логов фрагмент тела ответа.

        Args:
            response: Ответ сервера.
            limit: Максимальная длина фрагмента.

        Returns:
            str: Начало тела ответа без переводов строк.
        """
        try:
            return response.text[:limit].replace("\n", " ")
        except Exception:
            return "<нечитаемое тело ответа>"

    @staticmethod
    def _as_json(response: httpx.Response) -> dict[str, Any]:
        """Разбирает тело ответа как JSON-объект.

        Args:
            response: Ответ сервера.

        Returns:
            dict[str, Any]: Разобранный объект.

        Raises:
            TransportError: Если тело не является JSON-объектом.
        """
        try:
            data = response.json()
        except ValueError as exc:
            raise TransportError(f"ответ не является JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise TransportError(f"ожидался JSON-объект, получен {type(data).__name__}")
        return data

    # ----------------------------------------------------------------- чтение

    async def list_section_articles(self, section_id: int | None = None) -> list[dict]:
        """Возвращает список статей раздела, обходя все страницы.

        Args:
            section_id: Идентификатор раздела. По умолчанию берётся из
                конфигурации.

        Returns:
            list[dict]: Краткие карточки статей с полями ``id``, ``name``,
            ``updated_at``, ``section_id``, ``status``.

        Raises:
            TransportError: Если сервер не отдал список.
        """
        section = self._config.section_id if section_id is None else section_id
        articles: list[dict] = []
        page = 1
        while True:
            response = await self._request(
                "GET",
                f"api/knowledgebase/section/{section}/articles",
                params={"page": page, "per_page": self._config.per_page},
            )
            body = self._as_json(response)
            batch = body.get("data")
            if not isinstance(batch, list):
                raise TransportError(
                    f"раздел {section}: поле data отсутствует или не является списком"
                )
            articles.extend(item for item in batch if isinstance(item, dict))

            meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
            last_page = meta.get("last_page")
            if not isinstance(last_page, int) or page >= last_page or not batch:
                break
            page += 1

        logger.info("раздел %s: получено статей — %d", section, len(articles))
        return articles

    async def get_article(self, article_id: int | str) -> dict:
        """Загружает статью целиком.

        Args:
            article_id: Идентификатор статьи.

        Returns:
            dict: Объект статьи (содержимое поля ``data`` ответа) с полями
            ``id``, ``name``, ``content`` (HTML), ``section_id``,
            ``updated_at``.

        Raises:
            TransportError: Если ответ не содержит объекта статьи.
        """
        response = await self._request("GET", f"api/knowledgebase/article/{article_id}")
        body = self._as_json(response)
        data = body.get("data")
        if not isinstance(data, dict):
            raise TransportError(
                f"статья {article_id}: поле data отсутствует или не является объектом"
            )
        logger.debug(
            "статья %s загружена, размер содержимого %d симв.",
            article_id,
            len(str(data.get("content") or "")),
        )
        return data
