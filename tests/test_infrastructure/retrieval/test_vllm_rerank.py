"""Юнит-тесты переоценки через vLLM.

Проверяется разбор ответа ``/v1/rerank``, обработка ошибок и то, как клиент
встраивается в реранкер: сеть подменяет pytest-httpx.
"""

import json

import httpx
import pytest

from infrastructure.config.settings import Settings
from infrastructure.retrieval.rerank import (
    CrossEncoderReranker,
    RerankerHttpError,
    RerankerUnavailable,
    VLLMRerankerClient,
    build_reranker,
)
from infrastructure.retrieval.search import Chunk

BASE_URL = "http://vllm.test:8002"
RERANK_URL = f"{BASE_URL}/v1/rerank"
MODEL = "BAAI/bge-reranker-v2-m3"


def _results(*pairs: tuple[int, float]) -> dict:
    """Собирает ответ ``/v1/rerank``.

    Args:
        *pairs: Пары «индекс кандидата, оценка» в порядке ответа сервера.

    Returns:
        dict: Тело ответа сервера.
    """
    return {
        "id": "rerank-1",
        "model": MODEL,
        "usage": {"total_tokens": 8},
        "results": [
            {
                "index": index,
                "document": {"text": f"кандидат {index}"},
                "relevance_score": score,
            }
            for index, score in pairs
        ],
    }


def _chunk(number: int) -> Chunk:
    """Создаёт фрагмент для проверки порядка выдачи.

    Args:
        number: Номер фрагмента.

    Returns:
        Chunk: Фрагмент базы знаний.
    """
    return Chunk(
        chunk_id=f"c{number}",
        doc_code="СТО-1",
        doc_title="Документ",
        section="1",
        section_type="раздел",
        clause=None,
        text=f"кандидат {number}",
    )


@pytest.fixture
def client():
    """Создаёт клиент к подменённому серверу.

    Returns:
        VLLMRerankerClient: Клиент переоценки.
    """
    return VLLMRerankerClient(BASE_URL, model=MODEL, api_key="token-abc123")


class TestRequestShape:
    def test_endpoint_is_v1_rerank(self, client):
        """Адрес эндпоинта собирается из базового URL сервера."""
        assert client.endpoint == RERANK_URL

    def test_request_carries_query_documents_and_key(self, httpx_mock, client):
        """В запрос уходят модель, запрос, все кандидаты и ключ доступа."""
        httpx_mock.add_response(url=RERANK_URL, json=_results((0, 0.9), (1, 0.1)))
        client.predict("вопрос", ["кандидат 0", "кандидат 1"])
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        assert body["model"] == MODEL
        assert body["query"] == "вопрос"
        assert body["documents"] == ["кандидат 0", "кандидат 1"]
        # Без top_n сервер вернул бы не все оценки.
        assert body["top_n"] == 2
        assert request.headers["Authorization"] == "Bearer token-abc123"

    def test_no_authorization_without_key(self, httpx_mock):
        """Без ключа заголовок авторизации не шлётся."""
        httpx_mock.add_response(url=RERANK_URL, json=_results((0, 0.5)))
        VLLMRerankerClient(BASE_URL, model=MODEL).predict("вопрос", ["кандидат 0"])
        assert "Authorization" not in httpx_mock.get_requests()[0].headers

    def test_empty_candidates_do_not_hit_the_server(self, client):
        """Пустой список кандидатов не порождает запроса."""
        assert client.predict("вопрос", []) == []


class TestResponseParsing:
    def test_scores_return_in_order_of_candidates(self, httpx_mock, client):
        """Ответ отсортирован по убыванию оценки, порядок восстанавливается."""
        httpx_mock.add_response(
            url=RERANK_URL, json=_results((2, 0.9), (0, 0.5), (1, 0.1))
        )
        scores = client.predict("вопрос", ["ноль", "один", "два"])
        assert scores == [0.5, 0.1, 0.9]

    def test_short_answer_is_rejected(self, httpx_mock, client):
        """Оценок пришло меньше, чем кандидатов - это ошибка, а не выдача."""
        httpx_mock.add_response(url=RERANK_URL, json=_results((0, 0.9)))
        with pytest.raises(RerankerHttpError, match="ожидалось 2 оценок"):
            client.predict("вопрос", ["ноль", "один"])

    def test_index_out_of_range_is_rejected(self, httpx_mock, client):
        """Индекс вне списка кандидатов не молчит и не путает оценки."""
        httpx_mock.add_response(url=RERANK_URL, json=_results((5, 0.9), (0, 0.1)))
        with pytest.raises(RerankerHttpError, match="вне списка кандидатов"):
            client.predict("вопрос", ["ноль", "один"])

    def test_duplicate_index_is_rejected(self, httpx_mock, client):
        """Повтор индекса означает, что оценки есть не для всех кандидатов."""
        httpx_mock.add_response(url=RERANK_URL, json=_results((0, 0.9), (0, 0.1)))
        with pytest.raises(RerankerHttpError, match="не для всех кандидатов"):
            client.predict("вопрос", ["ноль", "один"])

    def test_non_numeric_score_is_rejected(self, httpx_mock, client):
        """Оценка не числом - ошибка разбора, а не падение при сортировке."""
        httpx_mock.add_response(
            url=RERANK_URL,
            json={
                "results": [
                    {"index": 0, "relevance_score": "высокая"},
                ]
            },
        )
        with pytest.raises(RerankerHttpError, match="не число"):
            client.predict("вопрос", ["ноль"])

    def test_not_json_is_rejected(self, httpx_mock, client):
        """Ответ не в JSON виден как ошибка сервиса."""
        httpx_mock.add_response(url=RERANK_URL, text="<html>502</html>")
        with pytest.raises(RerankerHttpError, match="не JSON"):
            client.predict("вопрос", ["ноль"])


class TestErrors:
    def test_missing_endpoint_says_what_to_start(self, httpx_mock, client):
        """404 объясняет, что реранкер обслуживает отдельный процесс vLLM."""
        httpx_mock.add_response(url=RERANK_URL, status_code=404, json={"error": "нет"})
        with pytest.raises(RerankerHttpError, match="vllm serve"):
            client.predict("вопрос", ["ноль"])

    def test_server_error_is_reported(self, httpx_mock, client):
        """Ошибка сервера доходит до вызывающего кода с кодом ответа."""
        httpx_mock.add_response(url=RERANK_URL, status_code=500, text="сломалось")
        with pytest.raises(RerankerHttpError, match="500"):
            client.predict("вопрос", ["ноль"])

    def test_unreachable_server_is_reported(self, httpx_mock, client):
        """Недоступный сервер виден как ошибка сервиса, а не как таймаут."""
        httpx_mock.add_exception(httpx.ConnectError("соединение отклонено"))
        with pytest.raises(RerankerHttpError, match="недоступен"):
            client.predict("вопрос", ["ноль"])


class TestRerankerIntegrationWithClient:
    """Готовый клиент встраивается в реранкер вместо хост-сервиса."""

    def test_hits_are_reordered_by_scores(self, httpx_mock):
        """Кандидаты пересортированы по оценкам vLLM."""
        httpx_mock.add_response(
            url=RERANK_URL, json=_results((2, 0.9), (0, 0.5), (1, 0.1))
        )
        reranker = CrossEncoderReranker(
            MODEL,
            candidates=3,
            client=VLLMRerankerClient(BASE_URL, model=MODEL),
        )
        hits = [(_chunk(0), 0.3), (_chunk(1), 0.2), (_chunk(2), 0.1)]
        result = reranker.rerank("вопрос", hits, limit=2)
        assert [chunk.chunk_id for chunk, _ in result] == ["c2", "c0"]

    def test_failure_falls_back_to_local_scores(self, httpx_mock, monkeypatch):
        """Сбой сервиса не роняет поиск: оценки считаются локально."""
        httpx_mock.add_exception(httpx.ConnectError("соединение отклонено"))
        reranker = CrossEncoderReranker(
            MODEL,
            candidates=2,
            client=VLLMRerankerClient(BASE_URL, model=MODEL),
        )
        monkeypatch.setattr(
            CrossEncoderReranker, "_local_scores", lambda self, q, texts: [0.1, 0.7]
        )
        hits = [(_chunk(0), 0.3), (_chunk(1), 0.2)]
        result = reranker.rerank("вопрос", hits, limit=2)
        assert [chunk.chunk_id for chunk, _ in result] == ["c1", "c0"]


class TestBuildReranker:
    """Выбор счётчика оценок делается настройкой."""

    def test_disabled_returns_nothing(self):
        """Выключенная переоценка не собирает клиентов."""
        assert build_reranker(Settings(rerank_enabled=False)) is None

    def test_vllm_provider_builds_vllm_client(self):
        """``vllm`` собирает клиент к ``/v1/rerank`` нужного сервера."""
        settings = Settings(
            rerank_enabled=True,
            rerank_provider="vllm",
            vllm_rerank_base_url=BASE_URL,
        )
        reranker = build_reranker(settings)
        assert isinstance(reranker._http, VLLMRerankerClient)
        assert reranker._http.endpoint == RERANK_URL

    def test_vllm_provider_falls_back_to_common_url(self):
        """Отдельный адрес реранкера не задан - берётся общий адрес vLLM."""
        settings = Settings(
            rerank_enabled=True, rerank_provider="vllm", vllm_base_url=BASE_URL
        )
        assert build_reranker(settings)._http.endpoint == RERANK_URL

    def test_local_provider_ignores_endpoint(self):
        """``local`` считает на месте, даже если адрес хост-сервиса задан."""
        settings = Settings(
            rerank_enabled=True,
            rerank_provider="local",
            rerank_endpoint="http://host.docker.internal:8899",
        )
        assert build_reranker(settings)._http is None

    def test_http_provider_requires_endpoint(self):
        """``http`` без адреса - понятная ошибка при сборке, а не при поиске."""
        settings = Settings(
            rerank_enabled=True, rerank_provider="http", rerank_endpoint=""
        )
        with pytest.raises(ValueError, match="RERANK_ENDPOINT"):
            build_reranker(settings)

    def test_auto_keeps_previous_behaviour(self):
        """``auto`` при заданном адресе идёт в хост-сервис, как раньше."""
        settings = Settings(
            rerank_enabled=True,
            rerank_endpoint="http://host.docker.internal:8899",
        )
        assert build_reranker(settings)._http is not None


class TestUnavailableEverywhere:
    """Когда не сработал ни сервис, ни локальный расчёт."""

    def test_error_names_both_reasons(self, httpx_mock, monkeypatch):
        """Сообщение называет обе причины, а не только отсутствие пакета.

        Пакета sentence-transformers в образе нет намеренно: оценки должен
        считать сервис, и молчать про его отказ нельзя.
        """
        httpx_mock.add_exception(httpx.ConnectError("соединение отклонено"))
        reranker = CrossEncoderReranker(
            MODEL, candidates=1, client=VLLMRerankerClient(BASE_URL, model=MODEL)
        )

        def _broken(self, query, texts):
            raise RerankerUnavailable("пакет sentence-transformers не установлен")

        monkeypatch.setattr(CrossEncoderReranker, "_local_scores", _broken)
        with pytest.raises(RerankerUnavailable) as info:
            reranker.rerank("вопрос", [(_chunk(0), 0.3)], limit=1)
        assert "не ответил" in str(info.value)
        assert "sentence-transformers" in str(info.value)


class TestRemoteCooldown:
    """Временный сбой не должен выключать переоценку навсегда.

    Раньше первая же ошибка выставляла клиент в None до перезапуска
    процесса: короткий сбой сети означал молчаливое падение качества с
    hit@1 0,829 до 0,561 на всё время работы приложения.
    """

    def test_remote_returns_after_cooldown(self, monkeypatch) -> None:
        """После паузы удалённый счётчик пробуется снова."""
        import infrastructure.retrieval.rerank as rr

        calls = {"n": 0}

        class _Flaky:
            def predict(self, query: str, texts: list[str]) -> list[float]:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise rr.RerankerHttpError("сеть моргнула")
                return [0.9] * len(texts)

        now = {"t": 1000.0}
        monkeypatch.setattr(rr.time, "monotonic", lambda: now["t"])

        reranker = rr.CrossEncoderReranker(client=_Flaky())
        reranker._local_scores = lambda q, t: [0.1] * len(t)

        chunks = [(_chunk(1), 1.0), (_chunk(2), 0.9)]

        reranker.rerank("вопрос", chunks, 2)
        assert calls["n"] == 1

        # Сразу после сбоя удалённый счётчик не трогаем.
        reranker.rerank("вопрос", chunks, 2)
        assert calls["n"] == 1

        # После паузы пробуем снова, и сервис возвращается в работу.
        now["t"] += rr.REMOTE_COOLDOWN_SECONDS + 1
        reranker.rerank("вопрос", chunks, 2)
        assert calls["n"] == 2
