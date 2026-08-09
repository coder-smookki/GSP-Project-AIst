# TESTS — тесты проекта

Описание test-набора: что где лежит, что покрывает, что требует живых
сервисов, как запускать в быстром цикле и полностью.

## Как запустить

Быстрый прогон без живых сервисов (по умолчанию в CI):

```sh
pytest -m "not llm and not embeddings and not qdrant"
```

Полный прогон требует поднятого Ollama и Qdrant:

```sh
pytest
```

или с отсечением только live-требующих:

```sh
pytest -m "not integration"
```

Настройки `[tool.pytest.ini_options]` в `pyproject.toml:47`:

- `asyncio_mode = "auto"` — async-тесты без явных декораторов;
- маркеры `integration`, `llm`, `embeddings`, `tools`, `corpus`, `qdrant`
  описаны там же (маркер `tools` объявлен, но в наборе не используется).

Линт и формат — `ruff` (config там же, `[tool.ruff]`): русскоязычным правилам
`RUF001-003` дано исключение.

## Что тестируется вживую

| Файл | Живой сервис |
| --- | --- |
| `test_ollama_server.py` | реальная LLM (Ollama): invoke/astream/tools |
| `test_ollama_embeddings.py` | реальная эмбеддинг-модель (Ollama) |
| `test_qdrant_live.py` | живой Qdrant (docker), но фиктивные 8-dim векторы — LLM не нужна |

Всё остальное — офлайн. `MockSearch` реализуется в
`infrastructure/retrieval/mock_search.py` и переэкспортируется через
`tests/fakes/fake_search.py`, чтобы его могли импортировать и прод-код, и тесты
(4-чаговый corpus in-memory, без Qdrant/Ollama).

### Особенности live-тестов

- `test_ollama_server.py` сам пропускается, если модель не загружена в Ollama;
- `test_qdrant_live.py` создаёт временную коллекцию и сам себя пропускает,
  если Qdrant недоступен — спокойно живёт в общем прогоне;
- конфиг-фикстура `conftest.py` только добавляет корень репозитория в
  `sys.path`.

## Карта тестов

### agent — `tests/test_agent/`

| Файл | Покрытие | Особенности |
| --- | --- | --- |
| `test_correction.py` | переписывание запроса (rewrite): обрезка кавычек, первая строка, отказ при длинном/неизменном, проглатывание ошибки модели, пустой вопрос без вызова модели | стаб-модель |
| `test_events.py` | конверт событий `emit`: допустимые kind → `{"aist":{"v":1,...}}`, неизвестный kind → ошибка, без стрима — no-op | настоящий `StateGraph` |
| `test_followups.py` | `build_followups` (dedup, лимит, fallback) и эмиссия событий через весь граф | `FakeModel`+`MockSearch`+`InMemoryStore`, `@asyncio` |
| `test_grading.py` | грейдер `grade_chunks`: keep-only-relevant, keep_partial, расхождение длин, отказ модели, схема `GradeResult` | fake structured-output |
| `test_memory_gate.py` | эвристики `mentions_user`/`worth_extracting`: маркеры первого лица, ё-фолдинг, границы слов, dict/объекты, игнор ассистентных сообщений | 10 тестов, параметризация |
| `test_numbers.py` | сверка чисел: выдуманные сроки, точное совпадение, окончания, игнор citation-маркеров, вставка предупреждения | чистая логика |
| `test_retrieve.py` | узел retrieve в составе графа: события до ответа, форма `retrieved`, sections-filter, `today_msk`, отказ модели, цикл rewrite | `@asyncio`, `MockSearch` |

### GUI контракт — `tests/test_gui/`

| Файл | Покрывает |
| --- | --- |
| `test_contract.py` | `build_input` (sections), `decode_aist_envelope` (v1/v2 чанк, `params`), `format_sources` (казённ/без clause), `pick_final_text` (приоритет ошибки, последний непустой) |

### Инфраструктура — `tests/test_infrastructure/`

| Файл | Покрывает | Сервис |
| --- | --- | --- |
| `config/test_settings.py` | дефолты `get_settings` (`llm_provider=="ollama-local"`), override по env (autouse lru-кэша) | none |
| `embeddings/test_ollama_embeddings.py` | `OllamaEmbeddingClient`: embed, embed_query, dim | **Ollama**, markers `embeddings`+`integration` |
| `ingest/test_pipeline.py` | офлайн-конвейер на синтетической фикстуре `kb_articles.json`: regex CLAUSE_HEAD, HTML-парсинг, деанонимия (по документу), чанкинг, стабильные id | none. |
| `ingest/test_real_corpus.py` | регрессию на реальном дампе `fixtures/corpus/` (20 статей, 44030): полнота, ≥1300 чанков, уникальные id, отсутствие плейсхолдеров, доля деанонимизации | marker `corpus`, `pytestmark` модуля |
| `llm/test_ollama_server.py` | живой `OllamaClient`: invoke, tools, astream | **Оllama**, marker `llm` |
| `llm/test_options.py` | опции клиента: на клиенте, reasoning off/on, num_ctx/repeat_penalty, tools через bind_tools, llama-cpp без extra_body | `FakeClient` |
| `orchestration/test_orchestration.py` | обёртки Prefect: dry-run чанки, flow/task реальные, `_document_ref_id`, сборка `_deployment_kwargs` из `AIST_*` | `importorskip("prefect")`, офлайн |
| `retrieval/test_qdrant_layer.py` | слой без Qdrant: BM25 по-русски, shared options, point id, payload-фильтры, `SearchUnavailable`, восстановление из payload | `client=None`, `_BrokenEmbedder` |
| `retrieval/test_qdrant_live.py` | гибрид на живом Qdrant: BM25, dense+sparse, стеммер, RRF, alias-переключение, sweep | **живой Qdrant**, markers `qdrant`+`integration` |
| `retrieval/test_search.py` | `MockSearch` и `to_citation` (регулярные/табличные, snippet ≤400) | in-memory |

### Корневые — `tests/`

| Файл | Покрывает |
| --- | --- |
| `test_evaluate.py` | хелперы `scripts/evaluate.py` (normalize, longest_common_run, chunk_contains, percentile) — импорт по пути из скрипта |
| `test_golden_set.py` | целостность golden-набора vs корпус: цитаты дословно, без плейсхолдеров, уникальность, ≥70% покрытия, оба типа не-ответов, ≥25% числовых | marker `corpus` |

## Fixtures

| Фикстура | Назначение |
| --- | --- |
| `tests/fixtures/kb_articles.json` | синтетический канон: намеренно воспроизводит дефекты исходника — обрезанный первый символ `.1.`, слияние/разбор clause'ов, оба порядка глоссария, один плейсхолдер с разными значениями |
| `tests/fixtures/corpus/` | 20 real-статей (id 88498–88551), секция 44030 (пожарная безопасность, дамп 08.08.2026) |
| `tests/fixtures/golden/questions.json` | golden-набор: `answerable` (`question`, `answer`, `quote`, `doc_id`, `clause`, `kind` ∈ числовой/фактический/процедурный) и `unanswerable` (`kind` ∈ вне области / в области но нет в корпусе, с причиной) |

## Чем руководствовалась структура тестов

- агентские тесты не трогают сеть: статы построены на `MockSearch` + стаб-модель;
- инфраструктурные тесты с риском сети помечены маркерами и в быстром прогоне
  отсеиваются;
- регрессии на реальном корпусе и golden-набор изолированы в `corpus` — там
  ход зависит от наличия фикстур (пропускаются, если файлов нет);
- офлайн-тесты пайплайна инджеда работают на синтетике, чтобы покрыть дефекты
  исходника без живых сервисов.