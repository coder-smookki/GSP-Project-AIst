# TESTS — тесты проекта

Описание test-набора: что где лежит, что покрывает, что требует живых
сервисов, как запускать в быстром цикле и полностью.

## Как запустить

Быстрый прогон без живых сервисов (по умолчанию в CI):

```sh
pytest -m "not integration"
```

Полный прогон требует поднятого Ollama и Qdrant:

```sh
pytest
```

или с отсечением только LLM/эмбеддингов (живой Qdrant сам себя пропускает,
если недоступен):

```sh
pytest -m "not llm and not embeddings"
```

Настройки `[tool.pytest.ini_options]` в `pyproject.toml:60`:

- `asyncio_mode = "auto"` — async-тесты без явных декораторов;
- маркеры `integration`, `llm`, `embeddings`, `tools`, `corpus`, `qdrant`,
  `vllm` описаны там же (маркер `tools` объявлен, но в наборе не
  используется).

Линт и формат — `ruff` (config там же, `[tool.ruff]`): русскоязычным правилам
`RUF001-003` дано исключение.

## Что тестируется вживую

| Файл | Живой сервис |
| --- | --- |
| `test_ollama_server.py` | реальная LLM (Ollama): invoke/astream/tools |
| `test_ollama_embeddings.py` | реальная эмбеддинг-модель (Ollama) |
| `test_qdrant_live.py` | живой Qdrant (docker), но фиктивные 8-dim векторы — LLM не нужна |
| `test_vllm_server.py` | живой сервер vLLM: генерация, эмбеддинги, переоценка |

Всё остальное — офлайн. `MockSearch` реализуется в
`infrastructure/retrieval/mock_search.py` и переэкспортируется через
`tests/fakes/fake_search.py`, чтобы его могли импортировать и прод-код, и тесты
(4-чаговый corpus in-memory, без Qdrant/Ollama).

### Особенности live-тестов

- `test_ollama_server.py` сам пропускается, если модель не загружена в Ollama;
- `test_qdrant_live.py` создаёт временную коллекцию и сам себя пропускает,
  если Qdrant недоступен — спокойно живёт в общем прогоне;
- `test_vllm_server.py` спрашивает у сервера `/v1/models` и пропускается с
  подсказкой, если сервера нет; адрес переоценки и эмбеддера берётся из
  своих настроек, поэтому группы пропускаются независимо;
- конфиг-фикстура `conftest.py` только добавляет корень репозитория в
  `sys.path`.

## Карта тестов

### agent — `tests/test_agent/`

| Файл | Покрытие | Особенности |
| --- | --- | --- |
| `test_planner.py` | планировщик: режим chat/kb, декомпозиция, бюджет подзапросов, запасной разбор JSON из текста при сбое структурированного вывода | стаб-модели |
| `test_search_agent.py` | ReAct-ветка: вызов инструмента, лимит раундов, переформулировка при пустой выдаче, `SearchUnavailable` | фейковый поиск + стаб-модель с tools |
| `test_retrieve.py` | поисковый путь в составе графа: chat-режим без поиска, параллельные подзапросы через `Send`, события до ответа, форма `retrieved`, sections-filter, `today_msk`, отказ модели | `@asyncio`, `MockSearch` |
| `test_merge_branches.py` | слияние ветвей в `finalize`: дедуп по `chunk_id`, круговое распределение мест между подзапросами, cap и нумерация | чистая логика |
| `test_turn_isolation.py` | изоляция ходов одного диалога: сброс накопительных каналов сигналом `RESET_CHANNEL`, отсутствие утечки источников между вопросами | настоящий граф, один тред |
| `test_state.py` | редьюсер `merge_or_reset`: сигнал сброса, суммирование вкладов веток | чистая логика |
| `test_events.py` | конверт событий `emit`: допустимые kind → `{"aist":{"v":1,...}}`, неизвестный kind → ошибка, без стрима — no-op | настоящий `StateGraph` |
| `test_followups.py` | `build_followups` (dedup, лимит, fallback) и эмиссия событий через весь граф | `FakeModel`+`MockSearch`+`InMemoryStore`, `@asyncio` |
| `test_stage_trace.py` | события этапов (`trace`) на реальном графе: пары start/done, поля при `done`, молчание дешёвого `condense` | `MockSearch` |
| `test_grading.py` | грейдер `grade_chunks`: keep-only-relevant, keep_partial, расхождение длин, отказ модели, схема `GradeResult` | fake structured-output |
| `test_memory_gate.py` | эвристики `mentions_user`/`worth_extracting`: маркеры первого лица, ё-фолдинг, границы слов, dict/объекты, игнор ассистентных сообщений | 10 тестов, параметризация |
| `test_numbers.py` | сверка чисел: выдуманные сроки, точное совпадение, окончания, игнор citation-маркеров, вставка предупреждения | чистая логика |
| `test_data_loss_guards.py` | регрессии потери данных: пустое резюме не стирает историю, пустой набор памяти не удаляет досье, авария поиска ≠ честный отказ | стабы |

### GUI контракт — `tests/test_gui/`

| Файл | Покрывает |
| --- | --- |
| `test_contract.py` | `build_input` (sections), `decode_aist_envelope` (v1/v2 чанк, `params`), `format_sources` (казённ/без clause), `parse_stage`, `pick_final_text` (приоритет ошибки, последний непустой) |
| `test_admin_login.py` | вход администратора из окружения, обязательность непустого пароля |
| `test_registration.py` | саморегистрация: страница `/register` отдаёт 404 без кода приглашения, регистрация по `REGISTRATION_CODE` |
| `test_chat_settings.py` | настройки областей знаний: нормализация MultiSelect, сохранение в сессии |
| `test_stages.py` | отрисовка этапов `StageView`: параллельные ветки различаются подзапросом, закрытие незавершённых шагов в `finally` |

### Инфраструктура — `tests/test_infrastructure/`

| Файл | Покрывает | Сервис |
| --- | --- | --- |
| `test_auth.py` | аутентификация aegra (`deploy/auth.py`): без токена — отказ, имя пользователя по узкому набору символов, изоляция ресурсов по владельцу | none |
| `test_content_app.py` | каталог базы знаний: `GET /knowledge/tree`, `GET /knowledge/documents/{doc_code}`, 404 на неизвестный код | none |
| `config/test_settings.py` | дефолты `get_settings` (`llm_provider=="ollama-local"`), override по env (autouse lru-кэша) | none |
| `config/test_env_discovery.py` | порядок чтения `.env`: корневой перекрывает `deploy/.env` (замер не должен брать чужие умолчания) | none |
| `embeddings/test_ollama_embeddings.py` | `OllamaEmbeddingClient`: embed, embed_query, dim | **Ollama**, markers `embeddings`+`integration` |
| `ingest/test_pipeline.py` | офлайн-конвейер на синтетической фикстуре `kb_articles.json`: regex CLAUSE_HEAD, HTML-парсинг, деанонимия (по документу), чанкинг, стабильные id, шаг QA-обогащения | none. |
| `ingest/test_qa_index.py` | QA-индексация: разбор ответа модели в вопросы, сборка `Q:/A:` текста, устойчивость к сбою вызова | стаб-модель |
| `ingest/test_incremental_state.py` | инкремент: отпечаток документа с версией конвейера и моделью эмбеддера, пропуск неизменившихся | none |
| `ingest/test_chunk_continuations.py` | склейка продолжений пунктов при разборе HTML | none |
| `ingest/test_real_corpus.py` | регрессию на реальном дампе `fixtures/corpus/` (20 статей, 44030): полнота, ≥1300 чанков, уникальные id, отсутствие плейсхолдеров, доля деанонимизации | marker `corpus`, `pytestmark` модуля |
| `llm/test_ollama_server.py` | живой `OllamaClient`: invoke, tools, astream | **Оllama**, marker `llm` |
| `config/test_model_registry.py` | сборка клиентов по провайдеру (ollama-local, llamacpp, vllm), оверрайды индексации, отказ на неизвестном провайдере с перечислением поддерживаемых | none |
| `embeddings/test_vllm_embeddings.py` | `VLLMEmbeddingClient`: разбор `/v1/embeddings`, текст вместо номеров токенов, заголовок авторизации, кеш размерности, ошибка 404 | `pytest-httpx` |
| `llm/test_vllm_client.py` | `VLLMClient`: тело запроса в терминах vLLM (`repetition_penalty`, `chat_template_kwargs`), разбор ответа и потока SSE, ошибка 401 | `pytest-httpx` |
| `llm/test_vllm_server.py` | живые клиенты vLLM: генерация, поток, эмбеддинги, переоценка | **живой vLLM**, markers `vllm`+`integration` |
| `llm/test_options.py` | опции клиента: на клиенте, reasoning off/on, num_ctx/repeat_penalty, tools через bind_tools, llama-cpp без extra_body | `FakeClient` |
| `orchestration/test_orchestration.py` | обёртки Prefect: dry-run чанки, flow/task реальные, `_document_ref_id`, сборка `_deployment_kwargs` из `AIST_*` | `importorskip("prefect")`, офлайн |
| `retrieval/test_qdrant_layer.py` | слой без Qdrant: BM25 по-русски, shared options, point id, payload-фильтры, `SearchUnavailable`, восстановление из payload | `client=None`, `_BrokenEmbedder` |
| `retrieval/test_qdrant_live.py` | гибрид на живом Qdrant: BM25, dense+sparse, стеммер, RRF, alias-переключение, sweep | **живой Qdrant**, markers `qdrant`+`integration` |
| `retrieval/test_kb_client.py` | HTTP-клиент базы знаний: вход, refresh, повторы, пагинация, маскирование секретов в ошибках | `pytest-httpx` |
| `retrieval/test_catalog.py` | каталог из индекса: дерево «область → документ → тип раздела», восстановление документа по `order_no`, кеш | in-memory |
| `retrieval/test_collection_dimension.py` | сверка размерности эмбеддера со схемой коллекции до первого запроса | in-memory |
| `retrieval/test_search.py` | `MockSearch` и `to_citation` (регулярные/табличные, snippet ≤400) | in-memory |
| `retrieval/test_vllm_rerank.py` | `VLLMRerankerClient`: разбор `/v1/rerank` с восстановлением порядка, ошибки сервера и разбора, выбор счётчика оценок настройкой `rerank_provider` | `pytest-httpx` |
| `retrieval/test_kb_live.py` | живая загрузка по API (нужны доступ и сеть), marker `integration` | **живой API** |

### Корневые и eval — `tests/eval/`

| Файл | Покрывает |
| --- | --- |
| `test_evaluate.py` | хелперы пакета `eval` (normalize, longest_common_run, chunk_contains, percentile) — импорт из `eval.golden` и `eval.metrics` |
| `test_golden_set.py` | целостность golden-набора vs корпус: цитаты дословно, без плейсхолдеров, уникальность, ≥70% покрытия, оба типа не-ответов, ≥25% числовых |
| `test_eval_contracts.py` | `Variant.identity`/`Environment.fingerprint` (детерминизм, чувствительность), раунд-трип `RunResult`, протоколы `Metric`/`Runner`/`Cache`/`StatTest`/`Reporter` |
| `test_eval_metrics.py` | hit@k/MRR/not_found/refusal/latency/answer_chars по известным наборам; registry |
| `test_eval_stats.py` | парные тесты: МакНемар на известной discordant-счет, Уилкоксон, пермутационный fallback, выбор теста по типу метрики, направление вердикта |
| `test_eval_cache.py` | детерминизм и чувствительность ключа кэша, file_hash, версия метрик, раунд-трип записи, метаданные и очистка |
| `test_eval_golden.py` | загрузка золотого набора: реальный фиксту, валидация структуры, limit, ошибки |
| `test_eval_runners.py` | `GoldenSetRunner`/`OperationalRunner` на фейках поиска и агента; метаданные и метрики |
| `test_eval_compare.py` | загрузка файла вариантов (список/объект, дубли, отсутствие), сборка payload |
| `test_eval_variants.py` | `make_settings` (орерайды), `build_variant` (search/agent/operационный); слоинг настроек (`test_variant_layering.py`) |
| `test_eval_cli.py` | дерево подкоманд `eval` (run/ab/bench/cache/report), управление кэшем |
| `test_eval_benchmark.py` | `main_async` замера: mock-защита, печать таблиц, запись JSON, пустой concurrency |
| `test_eval_report.py` | `metric_from_key`, группировка кэша в прогоны, рендер BENCHMARK.md, запись файла | marker `corpus` |
| `test_eval_tune.py` | перебор сетки параметров: лимит точек, сортировка по целевой метрике, кэш точек |

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