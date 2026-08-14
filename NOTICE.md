# Уведомление о собственности (NOTICE)

Этот файл соответствует разделу 4 лицензии Apache 2.0 и содержит сведения о
проекте и сторонних компонентах, которые распространяются вместе с ним.

## Проект

- **Название:** ?
- **Автор / правообладатель:** ?
- **Год первой публикации:** 2026
- **Лицензия кода проекта:** Apache License 2.0 (полный текст — `LICENSE.md`)

## Сторонние зависимости (Python)

Зависимости объявляются в `pyproject.toml`. Их лицензии определены из
инсталляции окружения с помощью `pip-licenses`. Краткая карта лицензий:

| SPDX-выражение                       | Примеры пакетов                                          | Статус |
|--------------------------------------|----------------------------------------------------------|--------|
| Apache-2.0                           | LangChain, LangChain-Ollama, LangChain-OpenAI, qdrant-client, OpenLLMetry/OpenInference, openai, safetensors, tokenizers, huggingface_hub, torch, transformers, sentence-transformers, prefect, langchain-\* | разрешающая, совместима |
| MIT / BSD-2-Clause / BSD-3-Clause / 0BSD / ISC / PSF-2.0 | SQLAlchemy, Pydantic, Pydantic-Core, httpx, rich, ruamel.yaml, joblib, click, Jinja2, MarkupSafe, PyYAML, typing_extensions, … | разрешающая, совместима |
| MPL-2.0                              | tqdm (всегда допускает MIT-вариант), certifi, pathspec (только dev, в ruff) | слабый копилефт; ASF считает MPL-2.0 совместимой с Apache 2.0 |
| **Artistic License ИЛИ GNU GPL v2+** | `text-unidecode` ← `python-slugify` ← `prefect`          | **двойная лицензия; копилефт** |

### text-unidecode — единственная копилефт-лицензия

`text-unidecode==1.3` распространяется по двойной лицензии
**GNU GPL v2+ ИЛИ Artistic License**. Она попадает в сборку транзитивно:
`prefect → python-slugify → text-unidecode`.

- **Где поставляется:** только образ оркестрации `deploy/Dockerfile.orchestration`
  (`pyproject` → extra `.[orchestration]`) и локальное dev-окружение / CI.
- **Где НЕ поставляется:** основной API-образ `deploy/Dockerfile` (устанавливает
  `.[rerank]`, `prefect` не подключается) и GUI-образ `deploy/Dockerfile.gui`
  (Chainlit не зависит от `python-slugify`/`text-unidecode` — проверено через
  `chainlit.Requires-Dist`).
- **Выбор для перераспространения:** для `text-unidecode` избрана
  **Artistic License** как допустимая альтернатива двойной лицензии; при таком
  выборе комбинация с кодом проекта (Apache 2.0) остаётся разрешающей.

## Сторонние зависимости в образах Docker

Помимо pip-пакетов, образы наследуют `python:3.12` (Debian, лицензия PSFL) и
в основном API-образе (`deploy/Dockerfile`) включают внешний пакет из PyPI
`aigra-api>=0.9.0,<0.10.0`. Лицензия `aigra-api` определяется его издателем
и не входит в данный репозиторий.

## Прочие активы

- `docs/demo/graph.png` — скриншот, созданный автором проекта; распространяется
  вместе с кодом проекта на той же лицензии.
- `deploy/clickhouse/config.xml` и `deploy/clickhouse/init.sh` — конфигурация
  ClickHouse и миграции БД автора проекта; Apache 2.0.
