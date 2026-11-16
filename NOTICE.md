# Уведомление о собственности (NOTICE)

Этот файл содержит сведения о проекте и сторонних компонентах, которые
распространяются вместе с ним.

## Проект

- **Название:** AIst - ИИ-поиск по корпоративной базе знаний
- **Авторы / правообладатели:** Ильин Кирилл Русланович (@coder-smookki), Пьянков Михаил (@Piankov-Michail)
- **Год первой публикации:** 2026
- **Лицензия кода проекта:** все права защищены (проприетарная лицензия, полный текст - `LICENSE`)

## Сторонние зависимости (Python)

Зависимости объявляются в `pyproject.toml`. Их лицензии определены из
инсталляции окружения с помощью `pip-licenses`. Краткая карта лицензий:

| SPDX-выражение                       | Примеры пакетов                                          | Статус |
|--------------------------------------|----------------------------------------------------------|--------|
| Apache-2.0                           | qdrant-client, chainlit, asyncpg, bcrypt, OpenLLMetry/OpenInference, openai, safetensors, tokenizers, huggingface_hub, transformers, sentence-transformers, prefect, grpcio | разрешающая, совместима |
| MIT                                  | LangChain, LangGraph, langchain-ollama, langchain-openai, SQLAlchemy, Pydantic, FastAPI, rich, PyJWT | разрешающая, совместима |
| MIT / BSD-2-Clause / BSD-3-Clause / 0BSD / ISC / PSF-2.0 | SQLAlchemy, Pydantic, Pydantic-Core, httpx, rich, ruamel.yaml, joblib, click, Jinja2, MarkupSafe, PyYAML, typing_extensions, … | разрешающая, совместима |
| MPL-2.0                              | certifi, bidict; двойные: tqdm (MPL-2.0 AND MIT), orjson (MPL-2.0 AND (Apache-2.0 OR MIT)); pathspec — только dev, в ruff | слабый копилефт пофайлово; ASF считает MPL-2.0 совместимой с Apache 2.0 |
| **Artistic License ИЛИ GNU GPL v2+** | `text-unidecode` ← `python-slugify` ← `prefect`          | **двойная лицензия; копилефт** |

### text-unidecode — единственная копилефт-лицензия

`text-unidecode==1.3` распространяется по двойной лицензии
**GNU GPL v2+ ИЛИ Artistic License**. Она попадает в сборку транзитивно:
`prefect → python-slugify → text-unidecode`.

- **Где поставляется:** только образ оркестрации `deploy/Dockerfile.orchestration`
  (`pyproject` → extra `.[orchestration]`) и локальное dev-окружение / CI.
- **Где НЕ поставляется:** основной API-образ `deploy/Dockerfile` (устанавливает
  проект без extras — `.[rerank]` и `prefect` не подключаются), GPU-образ
  реранкера `deploy/Dockerfile.reranker` (ставит только `sentence-transformers`,
  который `python-slugify`/`text-unidecode` не тянет) и GUI-образ
  `deploy/Dockerfile.gui` (Chainlit не зависит от `python-slugify`/`text-unidecode`
  — проверено через `chainlit.Requires-Dist`).
- **Выбор для перераспространения:** для `text-unidecode` избрана
  **Artistic License** как допустимая альтернатива двойной лицензии.

## Сторонние зависимости в образах Docker

Помимо pip-пакетов, образы наследуют `python:3.12-slim-bookworm` (Python под PSF-2.0,
базовая система Debian bookworm — набор пакетов под собственными свободными
лицензиями, преимущественно GPL/LGPL/MIT/BSD; распространяются как отдельные
системные пакеты, а не линкуются в код проекта) и
в основном API-образе (`deploy/Dockerfile`) включают внешний пакет из PyPI
`aegra-api>=0.9.0,<0.10.0`. Лицензия `aegra-api` определяется его издателем
и не входит в данный репозиторий.

## Сервисные образы стенда

Поднимаются как отдельные контейнеры, в код проекта не линкуются:

| Образ | Назначение | Лицензия |
|-------|-----------|----------|
| `qdrant/qdrant:v1.19.0` | векторный индекс | Apache-2.0 |
| `pgvector/pgvector:pg18` | база истории диалогов | PostgreSQL License (pgvector — PostgreSQL License) |
| `prefecthq/prefect:3.8.2-python3.12` | оркестрация индексации, только профиль orchestration | Apache-2.0 |
| `clickhouse/clickhouse-server` | хранилище трасс, только профиль obs | Apache-2.0 |
| `ghcr.io/openlit/openlit` | наблюдаемость, только профиль obs | Apache-2.0 |

## Модели

| Модель | Роль | Лицензия |
|--------|------|----------|
| `gpt-oss:20b` | генерация ответа | Apache-2.0 |
| `bge-m3` | векторное представление текста | MIT |
| `bge-reranker-v2-m3` | переоценка кандидатов | Apache-2.0 |

## Прочие активы

- `docs/demo/graph.png` — скриншот, созданный автором проекта; распространяется
  вместе с кодом проекта на той же лицензии.
- `deploy/clickhouse/config.xml` и `deploy/clickhouse/init.sh` — конфигурация
  ClickHouse и миграции БД автора проекта; Apache 2.0.
