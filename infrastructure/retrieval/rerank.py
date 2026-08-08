"""Переоценка кандидатов кросс-энкодером.

Зачем это нужно — коротко и с цифрами.

Гибридный поиск отдаёт кандидатов, отсортированных по слиянию RRF, но
**отделить релевантные от нерелевантных внутри одной выдачи он не может**.
Замер на корпусе: косинус dense-канала почти плоский по позициям — медиана
0,666 на первой против 0,613 на восьмой, разрыв 0,063, причём минимум первой
позиции (0,630) ниже максимума восьмой (0,681). Точки отсечения не
существует, поэтому порог тут бесполезен.

Кросс-энкодер видит запрос и текст **вместе**, а не по отдельности, и на тех
же данных разделяет начисто: 0,964 у подходящего фрагмента против 0,000 и
0,001 у неподходящих.

Почему не LLM-грейдер (issue #5). Замерено на qwen3:8b: оценка пяти чанков
одним вызовом со структурированным выводом стоит 10,8–23,3 с, а сокращение
контекста с 5 чанков до 3 экономит около 9 с. То есть грейдер тратит больше,
чем экономит, и делает это непредсказуемо: на одном вопросе он оставил все 5
из 5, на другом выбросил 4 из 5. Кросс-энкодер делает ту же работу за
доли секунды, потому что это маленькая специализированная модель, а не
генеративная.

Задержка (bge-reranker-v2-m3, MacBook 16 ГБ, MPS, синтетические пары):

=========  ===========  ===========
Кандидатов  fp32         fp16
=========  ===========  ===========
5           411 мс       427 мс
10          1558 мс      **880 мс**
30          5442 мс      1914 мс
=========  ===========  ===========

**Замер на золотом наборе: переоценка даёт много.** Ранее здесь стояло, что
гибридный поиск и так даёт ``hit@3 = 1.000`` и ``MRR 0.954``, а значит
переоценка почти ничего не покупает. Это оказалось неверно: те цифры сняты
на нескольких запросах вручную. На золотом наборе из 41 вопроса
(``scripts/evaluate.py``, боевой корпус, 1475 фрагментов) картина другая:

==================  ========  ========  ========  ========  ==========  =========
Режим               hit@1     hit@3     hit@5     MRR       не найдено  Поиск
==================  ========  ========  ========  ========  ==========  =========
без переоценки      0,561     0,707     0,780     0,648     9 из 41     163 мс
10 кандидатов       0,732     0,805     0,854     0,776     6 из 41     2673 мс
30 кандидатов       0,756     0,829     0,878     0,805     5 из 41     7795 мс
==================  ========  ========  ========  ========  ==========  =========

То есть переоценка поднимает hit@1 на 17 пунктов и вдвое сокращает число
вопросов, ответ на которые не доходит до модели вовсе. Рабочая точка — **10
кандидатов**: она забирает почти весь выигрыш тридцати при втрое меньшем
времени.

**Почему она всё-таки выключена по умолчанию.** Причина не в качестве, а в
поставке. Модель переоценки скачивается с huggingface.co при первом
обращении, а условие кейса — ноль обращений наружу, и проверяется оно
запуском без сети. Значит модель обязана лежать внутри образа, а вместе с
`torch` это добавляет к нему несколько гигабайт. Пока модель не уложена в
образ, включение сделало бы демонстрацию зависимой от интернета.

Чтобы включить осознанно, нужно три шага, а не один флаг:

1. ``uv sync --extra rerank`` — поставить `sentence-transformers`;
2. заранее выкачать модель на машину, где пойдёт демонстрация, чтобы
   `HF_HUB_OFFLINE=1` не ломал первый запрос;
3. ``RERANK_ENABLED=1`` и ``RERANK_CANDIDATES=10``.

Если включить флаг, не сделав первый шаг, поиск не сломается: пакет
отсутствует, `RerankerUnavailable` перехватывается в `QdrantSearch`,
переоценка отключается до перезапуска с записью в лог, а выдача остаётся в
порядке гибридного поиска.

Зависимость намеренно необязательная: `sentence-transformers` тянет torch,
а он не должен попадать в передаваемый образ, пока переоценка не включена.
Ставится отдельной группой: ``uv sync --extra rerank``.
"""

from __future__ import annotations

import logging
from typing import Any

from infrastructure.retrieval.search import Chunk

logger = logging.getLogger(__name__)

#: Модель переоценки. Apache-2.0, многоязычная (XLM-RoBERTa), работает с
#: русским без отдельной настройки.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerUnavailable(RuntimeError):
    """Переоценка запрошена, но недоступна."""


class CrossEncoderReranker:
    """Переоценивает кандидатов кросс-энкодером.

    Модель загружается лениво, при первом обращении: на слабой машине это
    десятки секунд, и платить их на старте приложения незачем, если
    переоценка так и не понадобится.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
        half_precision: bool = True,
        candidates: int = 10,
    ) -> None:
        """Создаёт переоценщик.

        Args:
            model_name: Имя модели кросс-энкодера.
            device: Устройство (``mps``, ``cuda``, ``cpu``); ``None`` —
                выбор библиотекой.
            max_length: Предел длины пары «запрос + фрагмент» в токенах.
            half_precision: Половинная точность. Даёт примерно трёхкратное
                ускорение на 30 кандидатах при неразличимом порядке выдачи.
            candidates: Сколько верхних кандидатов переоценивать. Остальные
                остаются в исходном порядке ниже — переоценивать всю
                предвыборку невыгодно: время растёт линейно, а нужное почти
                всегда в верхней части списка.
        """
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._half_precision = half_precision
        self._candidates = candidates
        self._model: Any | None = None

    def _load(self) -> Any:
        """Загружает модель при первом обращении.

        Returns:
            Any: Экземпляр ``CrossEncoder``.

        Raises:
            RerankerUnavailable: Если пакет не установлен или модель не
                загрузилась. Текст ошибки подсказывает, что делать.
        """
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailable(
                "переоценка включена, но пакет sentence-transformers не "
                "установлен: выполните `uv sync --extra rerank` либо "
                "выключите переоценку (RERANK_ENABLED=0)"
            ) from exc

        kwargs: dict[str, Any] = {"max_length": self._max_length}
        if self._device:
            kwargs["device"] = self._device
        if self._half_precision:
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        try:
            self._model = CrossEncoder(self._model_name, **kwargs)
        except Exception as exc:
            raise RerankerUnavailable(
                f"не удалось загрузить модель переоценки {self._model_name}: {exc}"
            ) from exc

        logger.info(
            "модель переоценки %s загружена (устройство %s, точность %s)",
            self._model_name,
            self._device or "по умолчанию",
            "половинная" if self._half_precision else "полная",
        )
        return self._model

    def rerank(
        self, query: str, hits: list[tuple[Chunk, float]], limit: int
    ) -> list[tuple[Chunk, float]]:
        """Переупорядочивает кандидатов по релевантности запросу.

        Args:
            query: Текст запроса пользователя.
            hits: Кандидаты в порядке гибридного поиска.
            limit: Сколько результатов вернуть.

        Returns:
            list[tuple[Chunk, float]]: Кандидаты по убыванию оценки
            кросс-энкодера. При любой ошибке возвращается исходный порядок,
            обрезанный до `limit`: переоценка — улучшение, а не условие
            работоспособности, и ронять из-за неё поиск нельзя.
        """
        if not hits:
            return []

        head = hits[: self._candidates]
        tail = hits[self._candidates :]

        try:
            model = self._load()
            scores = model.predict([(query, chunk.text) for chunk, _ in head])
        except RerankerUnavailable:
            raise
        except Exception as exc:
            logger.warning(
                "переоценка не выполнена, используем порядок поиска: %s", exc
            )
            return hits[:limit]

        ranked = sorted(
            (
                (chunk, float(score))
                for (chunk, _), score in zip(head, scores, strict=True)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        result = (ranked + tail)[:limit]
        logger.debug(
            "переоценено кандидатов %d, лучшая оценка %.3f",
            len(head),
            ranked[0][1] if ranked else 0.0,
        )
        return result


def build_reranker(settings) -> CrossEncoderReranker | None:
    """Создаёт переоценщик по настройкам.

    Args:
        settings: Настройки приложения.

    Returns:
        CrossEncoderReranker | None: Переоценщик или ``None``, если он
        выключен.
    """
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(
        settings.rerank_model,
        device=settings.rerank_device or None,
        half_precision=settings.rerank_half_precision,
        candidates=settings.rerank_candidates,
    )
