"""Клиент чат-модели на сервере vLLM.

vLLM отдаёт OpenAI-совместимый HTTP API: генерация - ``/v1/chat/completions``,
эмбеддинги - ``/v1/embeddings``, переоценка - ``/v1/rerank``. Авторизация,
если сервер запущен с ``--api-key``, идёт заголовком
``Authorization: Bearer <ключ>``, который проставляет сам OpenAI-совместимый
SDK. Поэтому клиент наследует общий :class:`OpenAICompatClient` и переопределяет
только дополнительные параметры запроса.

Чем vLLM отличается от Ollama в теле запроса:

* размер контекста задаётся при запуске сервера (``--max-model-len``), а не в
  запросе, поэтому ``num_ctx`` сюда не передаётся вовсе;
* штраф за повторы называется ``repetition_penalty`` и лежит в корне тела, а
  не внутри ``options`` (``options`` - расширение Ollama, vLLM его не знает);
* размышления Qwen3 выключаются через
  ``chat_template_kwargs={"enable_thinking": false}``.

Источник: https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server
и https://docs.vllm.ai/en/stable/features/reasoning_outputs
"""

from typing import Any

from .openai_compat import OpenAICompatClient

#: Ключ-заглушка для сервера, поднятого без ``--api-key``.
#:
#: Сам vLLM в этом режиме ключ не проверяет, но OpenAI-совместимый SDK
#: отказывается создавать клиент с пустым ключом. Значение взято из примеров
#: документации vLLM.
NO_KEY = "EMPTY"


class VLLMClient(OpenAICompatClient):
    """Чат-модель, обслуживаемая vLLM."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 300.0,
        repeat_penalty: float | None = None,
        reasoning: bool | None = False,
        temperature: float | None = 0.0,
        api_key: str = "",
    ):
        """Создаёт клиент.

        Args:
            model: Имя модели на сервере, например
                ``Qwen/Qwen3-8B``. У vLLM это идентификатор из ``--model``.
            base_url: Адрес сервера без суффикса ``/v1``.
            timeout: Предел ожидания ответа, с. По умолчанию большой:
                первый запрос может ждать прогрева сервера.
            repeat_penalty: Штраф за повторы; ``None`` - не слать, тогда
                действует умолчание сервера.
            reasoning: Режим размышлений. ``False`` - выключить
                (``enable_thinking=false``), ``True`` - включить, ``None`` -
                не вмешиваться в шаблон модели.
            temperature: Температура; ``None`` - умолчание сервера. Ноль
                нужен для стабильности отказов и структурированного вывода.
            api_key: Ключ доступа; пусто - подставляется заглушка.
        """
        super().__init__(
            model=model,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key or NO_KEY,
            temperature=temperature,
        )
        self._repeat_penalty = repeat_penalty
        self._reasoning = reasoning

    def _extra_body(self) -> dict | None:
        """Собирает дополнительные параметры запроса в формате vLLM.

        Returns:
            dict | None: Тело для ``extra_body`` либо ``None``, если слать
            нечего.
        """
        body: dict[str, Any] = {}
        if self._repeat_penalty is not None:
            body["repetition_penalty"] = self._repeat_penalty
        if self._reasoning is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self._reasoning)}
        return body or None
