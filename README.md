# ai_agent_project

Полное руководство по запуску, эксплуатации и диагностике — в [deploy/README.md](deploy/README.md).

Краткая версия:

```sh
cd deploy
cp .env.example .env
docker compose up -d                # базовый стек: postgres + aegra + gui
open http://localhost:8000          # GUI: admin / admin
```

Наблюдаемость LLM (OpenLIT, опционально):

```sh
cd deploy
echo 'OTLP_ENDPOINT=http://openlit:4318' >> .env
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile obs up -d
open http://localhost:3000          # OpenLIT UI: user@openlit.io / openlituser
```

## Структура проекта

Слоистая схема: `domain` (бизнес-правила) → `infrastructure` (адаптеры: LLM, векторный поиск, настройки) → `agent` (граф LangGraph) → `gui` (Chainlit-интерфейс). Запуск в Docker — из папки `deploy`.

```text
agent/                          # Ядро агента (LangGraph)
  graph.py                      #   граф: START→condense_context→retrieve→assistant; промпты агента
  state.py                      #   AgentState: сообщения, retrieved-выдержки, sections-фильтр, память
  events.py                     #   aist-конверт событий (custom): sources / followups / related / trace

domain/                         # Бизнес-правила (не зависят от LLM/БД)
  context/manager.py            #   суммаризация контекста (сжатие истории диалога)
  memory/schema.py              #   модели MemoryCollection / Memory для долгосрочной памяти

infrastructure/                 # Адаптеры и внешние сервисы
  config/
    settings.py                 #   Pydantic Settings (.env: LLM_PROVIDER, модели, URL Ollama)
    model_registry.py           #   фабрика моделей по провайдеру (ollama-local / vllm)
    composition_root.py         #   сборка приложения (внедрение зависимостей)
  llm/
    client.py                   #   Protocol-интерфейс LLM
    openai_compat.py            #   базовый клиент (num_ctx/repeat_penalty в extra_body)
    ollama_client.py            #   Ollama (num_ctx=8192, repeat_penalty=1.1)
    llamacpp_client.py          #   llama.cpp / vLLM (совместимый)
  embeddings/
    ollama_client.py            #   эмбеддинги Ollama
    llamacpp_client.py          #   эмбеддинги llama.cpp
  retrieval/search.py           #   поиск по базе знаний, формат Citation
  observability/tracer.py        #   OpenTelemetry/OpenLIT трассировка

gui/                            # Пользовательский интерфейс (Chainlit, порт :8000)
  chainlit_app.py               #   обработчики сообщений, stream через langgraph-sdk
  contract.py                   #   контракты: декодирование aist-конверта, цитирование, pick_final_text
  db.py                        #   БД GUI (пользователи, сессии)

deploy/                         # Запуск и эксплуатация (Docker)
  docker-compose.yml            #   базовый стек: postgres + aegra + gui
  docker-compose.observability.yml # профиль obs: openlit + clickhouse
  Dockerfile                    #   образ aegra
  Dockerfile.gui                #   образ gui
  aegra_entry.py                #   entry point LangGraph-платформы
  aegra.json                    #   конфигурация LangGraph-платформы
  clickhouse/init.sh            #   инициализация БД openlit
  README.md                     #   полное руководство по запуску/эксплуатации

scripts/                        # Утилиты командной строки
  cli_chat.py                   #   диалог с агентом из терминала (rich)
  server_cli.py                 #   запуск графа как локального LangGraph-сервера
  get_graph_visualization.py    #   сохранение Mermaid-схемы графа в graph.png

tests/                          # Тесты (pytest)
  fakes/fake_search.py          #   мок поиска (без внешних сервисов)
  test_agent/                   #   тесты графа и событий аист-конверта
  test_gui/                     #   тесты контрактов GUI
  test_infrastructure/          #   конфигурация, LLM-клиенты (юнит + интеграционные)
```

## Тестирование

Тесты командой `pytest`. Быстрые (юнит, без внешних сервисов) и медленные (интеграционные, с реальными моделями Ollama) разделены маркерами:

| Маркер | Тесты | Требования |
| --- | --- | --- |
| *(нет маркера)* | 51 юнит-тест | Ничего. Фейковые модели/поиск, посторонних сервисов не нужно |
| `llm` | 3 теста клиента | Запущенный Ollama с моделью `LLM_MODEL` (по умолчанию `qwen3.6:35b-a3b-nvfp4`) |
| `embeddings` | 3 теста эмбеддингов | Запущенный Ollama с моделью `EMBEDDING_MODEL` (`qwen3-embedding:8b`) |

### Быстрые тесты (без сервисов)

```sh
pytest -m "not llm and not embeddings"
```

### Полный прогон (нужен Ollama на хосте)

```sh
pytest
```

### Для GitHub Actions

- Быстрый job (PR check): `pytest -m "not llm and not embeddings"`.
- Отдельные job'ы, когда нужна полная проверка: поднять Ollama в сервис-контейнере, `pytest -m llm` и `pytest -m embeddings` отдельно.