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

**Что уже есть.** Собран и работает базовый каркас: Chainlit-интерфейс (:8000) общается с агентом на LangGraph через Aegra (:2026), модели живут в Ollama на хосте, история — в Postgres, поверх — опциональная трассировка OpenLIT. На этой базе выстроен RAG-контур: в графе появился узел `retrieve`, в состоянии — выбранные разделы (`sections`), найденные источники (`retrieved`) и подсказки (`followups`); агенту идут кастомные события (aist) и панель «Источники (N)», клики по follow-up работают без второго вызова LLM. Поиск пока синтетический (MockSearch) — ядро RAG, слой данных, ещё не подключён.

**Что дальше.** Следующий большой шаг — настоящий слой данных: свой парсер .docx + чанкинг (issue #11), деанонимизация (issue #12), загрузка в Qdrant с payload (§6.1, issue #13) и гибридный поиск dense+BM25+RRF (issue #14), подключённый за протоколом `Search` (issue #8). Параллельно GUI: стартеры-подсказки (#1), мультичаты и персистентность диалогов (#2, #3). Потом агентский уровень — Corrective RAG с грейдером и rewrite (#5) и адаптация к временным запросам (#7). Завершает план eval на золотом наборе (метрики ретрива и честных отказов) — до этого момента ретривер работает на фейке, потому что ему нужен реальный Qdrant.

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
  retrieval/search.py           #   протокол Search + формат Citation
  observability/tracer.py       #   OpenTelemetry/OpenLIT

gui/                            # Chainlit (:8000)
  chainlit_app.py               #   сообщения, стрим, панель источников, настройки
  contract.py                   #   aist-конверт, цитирование
  db.py                         #   БД GUI (пользователи)

deploy/                         # Docker: compose, Dockerfile, aegra_entry, README
scripts/                        # cli_chat, server_cli, get_graph_visualization
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
| *(нет маркера)* | 51 юнит-тест | Ничего. Фейковые модели/поиск, без сервисов |
| `llm` | 3 теста клиента | Ollama с `LLM_MODEL` |
| `embeddings` | 3 теста эмбеддингов | Ollama с `EMBEDDING_MODEL` |

- Быстро: `pytest -m "not llm and not embeddings"`
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

| Issue | Задача | Пункт | Слой |
| --- | --- | --- | --- |
| [#9](https://github.com/coder-smookki/GSP-Project-AIst/issues/9) | FilesSource — загрузка документов из папки | 5.4 | ingest |
| [#10](https://github.com/coder-smookki/GSP-Project-AIst/issues/10) | ApiSource — загрузка через API | 5.4 | ingest |
| [#11](https://github.com/coder-smookki/GSP-Project-AIst/issues/11) | docx-парсер + чанкинг атомов | 4.2–4.5 | ingest |
| [#12](https://github.com/coder-smookki/GSP-Project-AIst/issues/12) | Деанонимизация пофайловым словарём | 4.4 | ingest |
| [#13](https://github.com/coder-smookki/GSP-Project-AIst/issues/13) | Qdrant: коллекция + payload + загрузка | 6, 6.1 | retrieval |
| [#14](https://github.com/coder-smookki/GSP-Project-AIst/issues/14) | Qdrant: гибридный поиск dense+BM25+RRF | 6 | retrieval |
| [#8](https://github.com/coder-smookki/GSP-Project-AIst/issues/8) | Search facade — Qdrant за протоколом Search | 5.2–5.4 | retrieval |
| [#6](https://github.com/coder-smookki/GSP-Project-AIst/issues/6) | Реранкер (опциональный, RERANK_ENABLED) | D-07 | retrieval |
| [#5](https://github.com/coder-smookki/GSP-Project-AIst/issues/5) | Corrective RAG: grader + rewrite | 3.1, 7.2 | agent |
| [#7](https://github.com/coder-smookki/GSP-Project-AIst/issues/7) | Дата/время для временных запросов | 6.1 | agent |
| [#1](https://github.com/coder-smookki/GSP-Project-AIst/issues/1) | ChatStarter presets | 13.1–13.2 | gui |
| [#2](https://github.com/coder-smookki/GSP-Project-AIst/issues/2) | Персистентность диалогов | 3.2 | gui |
| [#3](https://github.com/coder-smookki/GSP-Project-AIst/issues/3) | Мультичаты + именование | 5.2 | gui |
| [#4](https://github.com/coder-smookki/GSP-Project-AIst/issues/4) | Fork + replay (опционально) | 12.1 | gui |
