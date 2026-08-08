# ai_agent_project

ИИ-поиск по корпоративной базе знаний (регламенты ИСМ.СМОЗиОБТ) с адаптивным RAG. Полностью on-premise: Chainlit → Aegra/LangGraph → Ollama → Postgres, векторный слой — Qdrant (в разработке, issues #13/#14).

Полное руководство по запуску и эксплуатации — [deploy/README.md](deploy/README.md).

```sh
cd deploy
cp .env.example .env
docker compose up -d                # базовый стек: postgres + aegra + gui
open http://localhost:8000          # GUI: admin / admin
```

Наблюдаемость (OpenLIT, опционально):

```sh
cd deploy
echo 'OTLP_ENDPOINT=http://openlit:4318' >> .env
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile obs up -d
open http://localhost:3000
```

## Резюме

**Что уже есть.** Работает базовый каркас: Chainlit-интерфейс (:8000) общается с агентом на LangGraph через Aegra (:2026), модели живут в Ollama на хосте, история — в Postgres, поверх — опциональная трассировка OpenLIT. В графе есть узел `retrieve`, в состоянии — выбранные разделы, найденные источники и подсказки; агент шлёт кастомные события и панель «Источники (N)».

Поверх этого собран **слой данных**: клиент к API базы знаний заказчика (вход, обновление токена, повторы), протокол `DataSource` с двумя реализациями (API и локальный каталог), разбор HTML-статей, пофайловая деанонимизация, чанкинг, схема Qdrant с гибридным поиском dense + BM25 и CLI индексации. На реальном корпусе раздела 44030 конвейер даёт **20 документов → 1446 чанков**.

**Что дальше.** Подключить `QdrantSearch` за фабрикой вместо заглушки (issue #8) и прогнать слой на живом контейнере Qdrant: серверный BM25 недоступен в локальном режиме клиента, поэтому гибридный поиск пока покрыт только тестами формы запроса. Затем Corrective RAG с грейдером (#5), реранкер (#6) и eval на золотом наборе. Параллельно GUI: стартеры (#1), мультичаты и персистентность (#2, #3).

## Интерфейс

Вход по паролю, история диалогов и список чатов — штатный слой данных Chainlit поверх того же Postgres, что у чекпоинтера агента. Идентификатор треда агента совпадает с идентификатором диалога, поэтому после перезагрузки страницы диалог продолжается с того же места без хранения соответствий.

- шесть стартовых подсказок по темам реального корпуса;
- выбор **областей знаний** (пожарная безопасность, охрана труда, ГО и ЧС, промышленная и производственная безопасность, работы повышенной опасности); пустой выбор означает поиск по всей базе;
- панель «Источники» открывается до первого токена ответа;
- подсказки следующего вопроса собираются из номеров пунктов найденных чанков, без второго вызова модели;
- самостоятельная регистрация выключена (`ALLOW_SELF_REGISTRATION=1` только для отладки).

## Адаптивность

Три механизма, каждый включён по результату замера, а не по названию кейса.

**Порог релевантности.** Если лучшее попадание слабее 0,55 по косинусу, выдача считается пустой и ассистент отвечает «В базе знаний информации нет». Без порога честный отказ механически недостижим: поиск всегда возвращает заданное число фрагментов, и на вопрос вне базы модель получила бы несколько настоящих пунктов регламента. Разделение чистое: релевантные запросы дают 0,68–0,72, посторонние 0,28–0,46.

**Коррекция запроса.** При пустой выдаче вопрос один раз переписывается языком регламента и поиск повторяется. Вызывается только при неудаче, поэтому средняя цена по всем запросам близка к нулю.

**Переоценка кросс-энкодером** (`RERANK_ENABLED`, выключена). Единственный механизм, способный отделить релевантное от нерелевантного *внутри* выдачи: счета dense-канала там почти плоские (0,666 против 0,613 между первой и восьмой позицией), а кросс-энкодер даёт 0,964 против 0,000. Выключена потому, что на текущем железе стоит 4,5 с и не сокращает контекст — решение пересматривается на демо-машине.

Чего сознательно нет: грейдера релевантности на каждый запрос. Замер показал 10–23 с на оценку пяти фрагментов при экономии 9 с — убыточно.

## Замеры на живом стенде

Снято 08.08.2026 на MacBook (Qdrant v1.19.0 в Docker, эмбеддер `bge-m3` в Ollama на хосте), корпус — 20 документов раздела 44030.

| Показатель | Значение |
| --- | --- |
| Документов → чанков | 20 → **1475** |
| Полная индексация | ~4 мин 40 с (узкое место — эмбеддинг, не загрузка) |
| Поиск целиком, медиана | **150–170 мс** |
| — из них эмбеддинг запроса | 139 мс |
| — из них Qdrant | ~11 мс |
| Первый (холодный) запрос | 160 мс |
| Размышления модели | выключены (`LLM_REASONING=0`): 16,3 с против 2,5 с при том же ответе |
| Русская морфология | падежи схлопываются: «средства/средству/средствами индивидуальной защиты» дают один и тот же счёт |
| TTFT (поиск + первый токен), 5 чанков | 16,6 с на qwen3:8b / 16 ГБ — упирается в чтение контекста, не в поиск |
| Размышления модели | выключены: 16,3 с против 2,5 с при том же ответе |
| Честный отказ | вопросы вне базы («столица Австралии», «курс доллара») дают отказ, вопросы по базе — ответ. Порог косинуса 0,55 |

Вклад поиска в TTFT — около 150 мс, и 93 % из них приходятся на обращение к модели эмбеддингов. Ускорять надо эмбеддер (батчи, `keep_alive`), а не индекс.

## Индексация корпуса

Корпус приходит из базы знаний заказчика по API либо из локального каталога. Выгрузка всех 20 статей раздела 44030 уже лежит в `tests/fixtures/corpus/`, поэтому проверить конвейер можно **без доступа к API и без поднятого Qdrant**:

```sh
uv run python scripts/ingest.py --dry-run --source files --corpus-dir tests/fixtures/corpus
```

Полная индексация в новое поколение коллекции с переключением алиаса:

```sh
export KB_EMAIL=... KB_PASSWORD=...          # в .env, в git не попадает
uv run python scripts/ingest.py --source api --generation 1 --switch-alias
```

| Флаг | Назначение |
| --- | --- |
| `--source api\|files` | источник корпуса |
| `--limit N` | обработать не более N документов |
| `--generation N` | писать в коллекцию `<алиас>_v<N>` |
| `--switch-alias` | переключить алиас после успешного прогона |
| `--dry-run` | разобрать и посчитать чанки, ничего не записывая |
| `--full` | полная переиндексация: не пропускать неизменившиеся документы |

**Повторный прогон инкрементальный.** Признак изменения — хеш разобранного документа вместе с версией конвейера, а не `updated_at` источника: у всех 20 статей выгрузки отметка времени попадает в один интервал массовой заливки и смысловых правок не отражает. Состояние читается из самого индекса (`content_hash` в payload), отдельного хранилища нет.

| Прогон | Результат | Время |
| --- | --- | --- |
| Первый | 20 документов, 1475 чанков | 2 мин 4 с |
| Повторный, без изменений | пропущено 20 | **1,76 с** |

Коды возврата: `0` — всё обработано, `1` — часть документов не обработана (подробности в отчёте), `2` — прогон не начался. Одна битая статья не роняет прогон, но **алиас не переключается, если были ошибки**: неполное поколение в прод не уезжает.

## Структура проекта

Слои: `domain` (бизнес-правила) → `infrastructure` (адаптеры: LLM, поиск, настройки) → `agent` (граф LangGraph) → `gui` (Chainlit). Запуск — из `deploy`.

```text
agent/                          # граф: START→condense_context→retrieve→assistant
  graph.py                      #   узлы, промпты агента, follow-ups
  state.py                      #   AgentState: messages, sections, retrieved, followups
  events.py                     #   aist-конверт custom-событий (sources/followups/...)

domain/
  context/manager.py            #   суммаризация истории диалога
  memory/schema.py              #   модели долгосрочной памяти

infrastructure/
  config/                       #   settings.py, model_registry.py, composition_root.py
  llm/                          #   клиенты: openai_compat, ollama, llamacpp
  embeddings/                   #   эмбеддинги: ollama, llamacpp
  ingest/                       # офлайн-конвейер построения индекса
    patterns.py                 #   регулярки разбора регламентов (единая точка правды)
    html_parse.py               #   HTML статьи -> разделы и атомы
    deanon.py                   #   пофайловый словарь [ОРГАНИЗАЦИЯ_N] -> расшифровка
    chunker.py                  #   атомы -> чанки, breadcrumb вместо overlap
    areas.py                    #   предметные области знаний (в источнике их нет)
    pipeline.py                 #   оркестрация прогона и отчёт
  retrieval/
    search.py                   #   протокол Search + формат Citation
    kb_client.py                #   HTTP-клиент базы знаний: вход, refresh, повторы
    source.py                   #   DataSource: ApiSource и FilesSource
    bm25.py                     #   опции серверного BM25 (одни для записи и запроса)
    store.py                    #   коллекция Qdrant, загрузка, уборка, поколения
    qdrant_search.py            #   гибридный поиск dense + BM25, слияние RRF
  observability/tracer.py       #   OpenTelemetry/OpenLIT

gui/                            # Chainlit (:8000)
  chainlit_app.py               #   сообщения, стрим, панель источников, настройки
  contract.py                   #   aist-конверт, цитирование
  db.py                         #   БД GUI (пользователи)

deploy/                         # Docker: compose, Dockerfile, aegra_entry, README
scripts/                        # ingest (индексация), cli_chat, server_cli, get_graph_visualization
tests/                          # pytest: fakes, test_agent, test_gui, test_infrastructure
docs/demo/graph.png             #   схема графа (генерация: scripts/get_graph_visualization.py)

.github/workflows/ci.yml        #   CI: юнит-тесты на каждый PR
.githooks/pre-commit            #   автоформатирование ruff (установка: scripts/setup_git_hooks.sh)
.vscode/                        #   настройки VS Code (ruff, formatOnSave)
```

## Сборка и разработка

Проект собирается через [uv](https://docs.astral.sh/uv/) (pyproject + закоммиченный `uv.lock`):

```sh
uv sync --extra dev             # рантайм + dev-инструменты (pytest, ruff, chainlit)
uv run pytest -m "not llm and not embeddings"   # быстрые юнит-тесты
uv run ruff check .
```

`--extra dev` обязателен — pytest/ruff лежат за этим extra. Версии фиксированы `uv.lock` (у всех и в CI одинаково); обновление — только через `uv lock`.

## Форматирование и стиль

Единый формат кода — **ruff==0.16.1** (зафиксирован в `uv.lock`, у всех байт-идентичный вывод).

**Автоматически при коммите.** Хук `.githooks/pre-commit` сам прогоняет `ruff format` по изменённым `.py` перед коммитом. Включить один раз:

```bash
bash scripts/setup_git_hooks.sh     # git config core.hooksPath .githooks
```

**Ручная проверка:**

```bash
uv run ruff format --check .        # формат
uv run ruff check .                 # линт (E/F/I/UP/B/SIM/ASYNC/RUF)
```

**VS Code** — расширение Ruff (`charliermarsh.ruff`) + `formatOnSave` из `.vscode/settings.json`; расширения рекомендованы в `.vscode/extensions.json`.

Настройки ruff (line-length 88, ignore-правила) — `pyproject.toml`, `[tool.ruff]`.

## Тестирование

| Маркер | Тесты | Требования |
| --- | --- | --- |
| *(нет маркера)* | 91 юнит-тест | Ничего. Фейковые модели/поиск, без сервисов и без сети |
| `corpus` | 11 регрессионных | Выгрузка корпуса в `tests/fixtures/corpus/` (в репозитории) |
| `llm` | 3 теста клиента | Ollama с `LLM_MODEL` |
| `embeddings` | 3 теста эмбеддингов | Ollama с `EMBEDDING_MODEL` |
| `qdrant` | пока нет | Живой сервер Qdrant; серверный BM25 недоступен в режиме `:memory:` |

- Быстро: `pytest -m "not llm and not embeddings and not corpus"`
- С регрессией на корпусе: `pytest -m "not llm and not embeddings"`
- Полный прогон (нужен Ollama на хосте): `pytest`
- CI на PR (`.github/workflows/ci.yml`): `uv sync --extra dev --frozen` + `pytest -m "not llm and not embeddings"`

## Текущее состояние проекта

**Готово (в main):**

| Пункт | Заголовок | Что реализовано |
| --- | --- | --- |
| 3.2 | Процессы, порты, где живут | Стек поднят: Chainlit :8000 → Aegra :2026 → Ollama :11434, Postgres-чекпоинтер: `deploy/docker-compose.yml` |
| 5.1 | Состояние графа | `AgentState`: `messages`, `summary`, `sections`, `retrieved`, `followups` — `agent/state.py` |
| 5.2 | Кастомные события: конверт и транспорт | aist-конверт v1 (trace/sources/related/followups), вещание в stream + state: `agent/events.py`, `gui/chainlit_app.py` |
| 5.3 | Схема цитаты | `Citation`/`to_citation` по полям (n, doc_code, clause, chunk_id, score, snippet): `infrastructure/retrieval/search.py` |
| 7.1 | Профили моделей | Слоты fast/quality (D-27): `LLM_SUPPORT_MODEL` + реестр `infrastructure/config/model_registry.py` |
| 9.3 | Наблюдаемость: расщепление по лицензиям | OpenLIT/OTLP только в overlay-файле (obs), в базовом стеке нет: `deploy/docker-compose.observability.yml` |
| 9.6 | Тесты графа без Ollama | MockSearch/FakeEmbedding, юнит-тесты: `tests/`, маркеры `llm`/`embeddings` |
| 13.2 | Демо-сценарий (частич.) | Панель источников до первого токена + follow-up из пунктов чанков: `gui/chainlit_app.py` |

Дополнительно: CI на PR (`pytest -m "not llm and not embeddings"`), ruff==0.16.1 + pre-commit hook, `uv.lock` закоммичен.

**В работе (issues):**

| Issue | Задача | Пункт | Слой | Статус |
| --- | --- | --- | --- | --- |
| [#9](https://github.com/coder-smookki/GSP-Project-AIst/issues/9) | FilesSource — загрузка документов из папки | 5.4 | ingest | частично: источник готов, разбор .docx остаётся |
| ~~[#10](https://github.com/coder-smookki/GSP-Project-AIst/issues/10)~~ | ApiSource — загрузка через API | 5.4 | ingest | **готово** |
| [#11](https://github.com/coder-smookki/GSP-Project-AIst/issues/11) | docx-парсер + чанкинг атомов | 4.2–4.5 | ingest | частично: HTML-парсер и чанкинг готовы |
| ~~[#12](https://github.com/coder-smookki/GSP-Project-AIst/issues/12)~~ | Деанонимизация пофайловым словарём | 4.4 | ingest | **готово** |
| [#13](https://github.com/coder-smookki/GSP-Project-AIst/issues/13) | Qdrant: коллекция + payload + загрузка | 6, 6.1 | retrieval | реализовано, ждёт живого сервера |
| [#14](https://github.com/coder-smookki/GSP-Project-AIst/issues/14) | Qdrant: гибридный поиск dense+BM25+RRF | 6 | retrieval | реализовано, ждёт живого сервера |
| [#8](https://github.com/coder-smookki/GSP-Project-AIst/issues/8) | Search facade — Qdrant за протоколом Search | 5.2–5.4 | retrieval |
| [#6](https://github.com/coder-smookki/GSP-Project-AIst/issues/6) | Реранкер (опциональный, RERANK_ENABLED) | D-07 | retrieval |
| [#5](https://github.com/coder-smookki/GSP-Project-AIst/issues/5) | Corrective RAG: grader + rewrite | 3.1, 7.2 | agent |
| [#7](https://github.com/coder-smookki/GSP-Project-AIst/issues/7) | Дата/время для временных запросов | 6.1 | agent |
| [#1](https://github.com/coder-smookki/GSP-Project-AIst/issues/1) | ChatStarter presets | 13.1–13.2 | gui |
| [#2](https://github.com/coder-smookki/GSP-Project-AIst/issues/2) | Персистентность диалогов | 3.2 | gui |
| [#3](https://github.com/coder-smookki/GSP-Project-AIst/issues/3) | Мультичаты + именование | 5.2 | gui |
| [#4](https://github.com/coder-smookki/GSP-Project-AIst/issues/4) | Fork + replay (опционально) | 12.1 | gui |
