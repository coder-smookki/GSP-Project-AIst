"""Настройки приложения.

Значения читаются из окружения и файла ``.env``. Секреты (пароль к базе
знаний) в репозиторий не попадают: ``.env`` в ``.gitignore``.
"""

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Провайдеры моделей, которые реестр умеет собирать (см. model_registry).
MODEL_PROVIDERS = ("ollama-local", "llamacpp", "vllm")

#: Способы посчитать переоценку кандидатов.
#:
#: ``auto`` - хост-сервис, если задан ``rerank_endpoint``, иначе локальный
#: расчёт (историческое поведение); ``local`` - только локальный
#: кросс-энкодер; ``http`` - только хост-сервис ``deploy/rerank_server.py``;
#: ``vllm`` - эндпоинт ``/v1/rerank`` сервера vLLM.
RERANK_PROVIDERS = ("auto", "local", "http", "vllm")


class Settings(BaseSettings):
    """Конфигурация всех слоёв: модели, поиск, источник корпуса."""

    # ------------------------------------------------------------------ LLM
    llm_provider: str = "ollama-local"  # ollama-local | llamacpp | vllm
    llm_model: str = "qwen3.6:35b-a3b-nvfp4"
    llm_support_model: str = "None"
    llm_num_ctx: int = 16384
    llm_repeat_penalty: float = 1.1

    #: Размышления модели перед ответом.
    #:
    #: Выключены по умолчанию: на qwen3:8b они стоят шестикратного времени
    #: ответа (16,3 с против 2,5 с на одном и том же вопросе) при том же
    #: тексте ответа. Скорость входит в оценку жюри напрямую.
    #: Включать только если замер покажет выигрыш в качестве.
    llm_reasoning: bool = False

    ollama_local_base_url: str = "http://localhost:11434"
    llamacpp_base_url: str = "http://localhost:8080"
    vllm_base_url: str = "http://localhost:8000"

    # ------------------------------------------------------------------ vLLM
    #: Ключ доступа к vLLM (сервер запущен с ``--api-key``).
    #:
    #: Уходит заголовком ``Authorization: Bearer <ключ>``. Пусто - сервер
    #: поднят без ключа; клиент всё равно шлёт заглушку, потому что
    #: OpenAI-совместимый SDK отказывается работать с пустым ключом.
    vllm_api_key: str = Field(default="", repr=False)

    #: Предел ожидания ответа vLLM, с. Значение большое по той же причине,
    #: что и у Ollama: первый запрос может ждать прогрева сервера.
    vllm_timeout: float = 300.0

    #: Адрес vLLM с эмбеддером. Пусто - берётся ``vllm_base_url``.
    #:
    #: Один процесс vLLM обслуживает ровно одну модель, поэтому генерация,
    #: эмбеддинги и переоценка живут на разных портах.
    vllm_embedding_base_url: str = ""

    #: Адрес vLLM с моделью переоценки. Пусто - берётся ``vllm_base_url``.
    vllm_rerank_base_url: str = ""

    # ----------------------------------------------------------- эмбеддинги
    embedding_provider: str = "ollama-local"  # ollama-local | llamacpp | vllm
    embedding_model: str = "bge-m3"

    #: Инструкция, добавляемая к тексту ЗАПРОСА перед кодированием.
    #:
    #: Нужна моделям семейства qwen3-embedding: их карточка прямо
    #: предупреждает, что без инструкции на стороне запроса качество поиска
    #: падает. Для bge-m3 инструкция не нужна и поле остаётся пустым —
    #: у неё запрос и документ кодируются одинаково.
    embedding_query_instruction: str = ""

    # ------------------------------------------ наблюдаемость (OpenLIT/OTLP)
    otlp_endpoint: str = ""  # напр. http://openlit:4318; пусто = трассировка выключена

    # ---------------------------------------------------------------- поиск
    retrieval_provider: str = "mock"
    use_mock_search: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_timeout: float = 10.0

    qdrant_collection: str = "aist_chunks"

    retrieval_top_k: int = 5

    retrieval_prefetch: int = 30

    retrieval_rrf_k: int = 60

    retrieval_min_relevance: float = 0.55

    #: Сколько раз переписывать запрос при пустой выдаче (0 = без коррекции).
    retrieval_max_rewrites: int = 1

    #: Общий бюджет обращений к поиску на один запрос пользователя.
    #:
    #: Ограничивает сверху сумму поисковых вызовов по всем веткам-агентам
    #: (план + повторы). Число подзапросов планировщика выводится отсюда:
    #: ``retrieval_max_requests // (retrieval_branch_retries + 1)``.
    retrieval_max_requests: int = 10

    #: Сколько раз веточный агент может повторить поиск (ReAct-петля).
    #: ``0`` — без повторов, один поисковый вызов на ветку.
    retrieval_branch_retries: int = Field(default=1, ge=0)

    # ------------------------------------------------------- переоценка
    rerank_enabled: bool = False

    #: Кто считает переоценку: auto | local | http | vllm.
    #:
    #: ``auto`` сохраняет прежнее поведение: есть ``rerank_endpoint`` -
    #: хост-сервис, нет - локальный кросс-энкодер.
    rerank_provider: str = "auto"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = ""
    rerank_half_precision: bool = True

    #: URL хост-сервиса переоценки (deploy/rerank_server.py), например
    #: http://host.docker.internal:8899. Colima не пробрасывает GPU, и инференс
    #: кросс-энкодера в контейнере в ~16 раз медленнее нативного macOS-торча
    #: (MPS). При пустом значении реранкер считается локально, как раньше.
    rerank_endpoint: str = ""

    rerank_candidates: int = 10

    # ---------------------------------------------------------- грейдинг
    grading_enabled: bool = False

    grading_keep_partial: bool = True

    # ------------------------------------------------------ сверка чисел
    verify_numbers: bool = True

    #: Размер батча загрузки точек в Qdrant.
    ingest_batch_size: int = 128

    # ------------------------------------------------- источник корпуса (API)
    kb_base_url: str = "https://hihub.ru/"
    kb_email: str = ""
    kb_password: str = Field(default="", repr=False)
    kb_section_id: int = 44030
    kb_token_name: str = "aist"
    kb_per_page: int = 100
    kb_max_retries: int = 3

    #: Каталог локальной выгрузки корпуса для `FilesSource`.
    corpus_dir: str = "tests/fixtures/corpus"

    #: Офлайн-режим передаваемого образа. При ``True`` источники, ходящие в
    #: сеть, не инициализируются вовсе — «0 внешних API» блокирующий критерий
    #: кейса, и нарушить его молча нельзя.
    aist_offline: bool = False

    # Читаются оба файла, причём корневой `.env` перекрывает `deploy/.env`.
    #
    # Зачем два. Настройки стенда живут в `deploy/.env`, потому что оттуда их
    # берёт docker compose. Но пути в pydantic считаются от текущего каталога,
    # и всё, что запускается с хоста из корня репозитория (замер качества,
    # индексация, scripts/verify.sh), этот файл просто не видело. Настройки
    # молча брались из умолчаний ниже, и человек был уверен, что померил свою
    # конфигурацию, хотя мерил чужую: выставленные в `deploy/.env` prefetch=10,
    # rrf=40 и min_rel=0.45 не применялись, работали 30, 60 и 0.55.
    #
    # Сетевые адреса (QDRANT_URL, OLLAMA_LOCAL_BASE_URL, POSTGRES_HOST) в
    # `deploy/.env` держать не нужно: внутри стека их подставляет сам compose,
    # а на хосте верны умолчания ниже. В шаблоне они закомментированы.
    model_config = SettingsConfigDict(
        env_file=("deploy/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("retrieval_provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"mock", "qdrant"}:
            raise ValueError(
                f"неизвестная реализация поиска {value!r}: ожидается mock или qdrant"
            )
        return normalized

    @field_validator("llm_provider", "embedding_provider")
    @classmethod
    def _check_model_provider(cls, value: str, info: ValidationInfo) -> str:
        """Проверяет, что провайдер моделей реализован в реестре.

        Args:
            value: Значение настройки провайдера.
            info: Контекст валидации; нужен ради имени поля в сообщении.

        Returns:
            str: Нормализованное значение провайдера.

        Raises:
            ValueError: Если провайдер не входит в ``MODEL_PROVIDERS``.
        """
        normalized = value.strip().lower()
        if normalized not in MODEL_PROVIDERS:
            raise ValueError(
                f"неизвестный провайдер {info.field_name}={value!r}: "
                f"поддерживаются {', '.join(MODEL_PROVIDERS)}"
            )
        return normalized

    @field_validator("rerank_provider")
    @classmethod
    def _check_rerank_provider(cls, value: str) -> str:
        """Проверяет способ переоценки кандидатов.

        Args:
            value: Значение настройки ``rerank_provider``.

        Returns:
            str: Нормализованное значение.

        Raises:
            ValueError: Если значение не входит в ``RERANK_PROVIDERS``.
        """
        normalized = value.strip().lower()
        if normalized not in RERANK_PROVIDERS:
            raise ValueError(
                f"неизвестный способ переоценки {value!r}: "
                f"поддерживаются {', '.join(RERANK_PROVIDERS)}"
            )
        return normalized

    @field_validator(
        "retrieval_top_k",
        "retrieval_prefetch",
        "ingest_batch_size",
        "retrieval_max_requests",
    )
    @classmethod
    def _check_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"значение должно быть положительным, получено {value}")
        return value

    def resolved_retrieval_provider(self) -> str:
        return "mock" if self.use_mock_search else self.retrieval_provider

    def resolved_vllm_embedding_base_url(self) -> str:
        """Возвращает адрес vLLM для эмбеддингов.

        Returns:
            str: ``vllm_embedding_base_url``, а при пустом значении -
            ``vllm_base_url``.
        """
        return self.vllm_embedding_base_url or self.vllm_base_url

    def resolved_vllm_rerank_base_url(self) -> str:
        """Возвращает адрес vLLM для переоценки.

        Returns:
            str: ``vllm_rerank_base_url``, а при пустом значении -
            ``vllm_base_url``.
        """
        return self.vllm_rerank_base_url or self.vllm_base_url


@lru_cache
def get_settings() -> Settings:
    """Возвращает единственный экземпляр настроек.

    Returns:
        Settings: Настройки приложения; результат кешируется.
    """
    return Settings()
