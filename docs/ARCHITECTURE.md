# ARCHITECTURE — общая архитектура проекта

Верхнеуровневый обзор всех слоёв системы: от вопроса в интерфейсе до хранения
индекса и автоматизации индексации. Глубина по каждому слою — в профильных
документах:

| Слой | Где описан |
| --- | --- |
| Граф ассистента | `docs/AGENT.md` |
| Поиск и retrieve | `docs/RETRIEVE.md` |
| Индексация корпуса и Prefect | `docs/INGEST.md` |
| Docker-стенд, запуск, эксплуатация | `docs/DEPLOY.md` |
| Интерфейс Chainlit и контракт событий | `docs/GUI.md` |

---

## 1. Карта системы

```mermaid
flowchart LR
    U[Пользователь<br/>браузер] -->|:8000| G[gui · Chainlit]
    G -->|REST /threads/.../runs/stream<br/>+ aist-конверт| A[aegra · LangGraph<br/>:2026]

    A -->|SQL / pgvector| PG[(postgres<br/>:5432)]
    A -->|LLM-запросы| OLL[(Ollama на хосте<br/>:11434)]

    subgraph Agent [Уровень агента]
        A
    end

    A -->|search.query| Q[(qdrant<br/>:6333)]
    Q --> QI[Индекс: dense + BM25 + RRF]

    subgraph Ingest [Уровень загрузки]
        SRC[Источник корпуса<br/>API базы знаний / локальный каталог]
        P[Конвейер ingest<br/>разбор → деанон → чанки → области → эмбеддинги]
    end

    SRC --> P
    P -->|upsert + sweep_document| Q
    Q --> QI[Индекс: dense + BM25 + RRF]

    subgraph Prefect [Оркестрация]
        PF[prefect-server :4200<br/>UI, деплоймент aist-ingest]
        PR[prefect-runner · aist-ingest]
    end
    PF <-->|API| PR
    PR --> P

    subgraph Obs [Наблюдаемость]
        OL[openlit :3000]
        CH[(clickhouse)]
    end
    A -.->|OTLP :4318| OL
    OL --> CH
```

Три потока данных:

1. **Запрос (online).** Пользователь → Chainlit → aegra (граф LangGraph) →
   Qdrant (поиск) + Ollama (генерация) + Postgres (чекпоинтеры и история).
   Результат с панелью источников возвращается в GUI.
2. **Индексация (offline).** Источник корпуса → конвейер ingest → Qdrant.
   Запускается CLI-скриптом `scripts/ingest.py` либо деплойментом Prefect
   `aist-ingest`.
3. **Наблюдаемость (опционально).** Трассы OTLP из aegra → OpenLIT → ClickHouse.

---

## 2. Уровень агента (agent/)

Агент — граф LangGraph, собираемый `create_app_agent` (`agent/graph.py:128`).
Точка входа в рантайме — REST-канал LangGraph-сервера aegra
(`/threads/{id}/runs/stream`), а сама сборка и связывание с поиском, моделями и
памятью — в `infrastructure/config/composition_root.py:12`
(`create_application`).

Генеративная схема графа лежит в `docs/demo/graph.png` (пересоздаётся скриптом
`scripts/get_graph_visualization.py`):

<img src="./demo/graph.png" alt="Схема графа ассистента" width="480"/>

Тот же граф в разрезе узлов:

```mermaid
flowchart TD
    S([START])
    S --> R[reset_turn<br/>graph.py:144]
    R --> CC[condense_context<br/>graph.py:284]
    CC --> RV[retrieve<br/>graph.py:214]

    subgraph Loop [Цикл по чанкам: пустая выдача → переписать]
        RV -- "needs_rewrite" --> RQ[rewrite_query<br/>graph.py:273]
        RQ --> RV
    end

    RV --> AS[assistant<br/>graph.py:152]
    AS --> WM{should_extract_memory<br/>graph.py:109}
    WM -- есть маркеры первого лица --> WR[write_memory<br/>graph.py:303]
    WM -- нет --> E([END])
    WR --> E([END])
```

Роли узлов и ссылки на строки — в `docs/AGENT.md`; здесь — схема потока:
`reset_turn` → `condense_context` → `retrieve` ⇄ `rewrite_query` (при пустой выдаче)
→ `assistant` → `write_memory` (только по маркерам первого лица).

Состояние — `AgentState` (`agent/state.py:7`): `messages` (редьюсер
`add_messages`) + `summary`, `retrieved`, `followups`, `areas`, `query` и
служебные флаги.

### Как retrieve встраивается в агента

Узел `retrieve` (graph.py:214) не знает, что стоит за протоколом `Search` —
Qdrant или заглушка; фабрика `get_search` (`factory.py:13`) подставляет
реализацию по `RETRIEVAL_PROVIDER`. Три исхода поиска («найдено / пусто /
`SearchUnavailable`») и цикл коррекции пустой выдачи описаны в `docs/RETRIEVE.md`
(разделы 2–3).

### Что делает грейдер (и почему он выключен)

`grading.py:30` (`grade_chunks`) — отсев фрагментов **моделью** перед
генерацией: одной оценкой на всю выдачу модель ставит каждому фрагменту вердикт
«релевантно / частично / нет». Реализован и замерен на полном пути ответа, но
**выключен** (`GRADING_ENABLED=0`): на текущем корпусе честный отказ он не
покупает (его делает порог 0.55 до грейдера), а с отсевом «частично» теряет
нужные фрагменты. Оставлен заготовкой под другой корпус. Детали и цифры —
`docs/RETRIEVE.md`, раздел 10.

### Как работает реранкер (включён)

Реранкер — кросс-энкодер `BAAI/bge-reranker-v2-m3` (`rerank.py:105`), **не узел
графа**, а часть `QdrantSearch`: retrieve берёт `RERANK_CANDIDATES=10`, реранкер
переранжирует, после чего — порог релевантности. Подробности, замеры и
офлайн-обёртка весов в образ — в `docs/RETRIEVE.md` (раздел 8).

---

## 3. Уровень retrieve: как получаются чанки

Внутреннее устройство узла и поискового слоя (`infrastructure/retrieval/`):

```mermaid
flowchart TB
    RV[retrieve · agent/graph.py:214]
    RV -->|search.query query, areas, sections| P{Протокол Search}

    P -->|qdrant| QS[QdrantSearch · qdrant_search.py:20]
    P -->|mock| MS[MockSearch · заглушка для тестов]

    subgraph Hybrid [Гибридный запрос · qdrant_search.py:102]
        E[Эмбеддер bge-m3] -->|dense-вектор| P1[Prefetch dense<br/>filter · limit prefetch]
        BM[BM25-документ, серверный] -->|sparse| P2[Prefetch bm25<br/>filter · limit prefetch]
        P1 --> RRF[RrfQuery · RRF k=60]
        P2 --> RRF --> TOP[fetch limit кандидатов]
    end

    QS --> Hybrid

    TOP --> RR{Реранкер включён?}
    RR -- да --> RK[Переоценка 10 кандидатов<br/>bge-reranker-v2-m3]
    RK --> K[top_k фрагментов]
    RR -- нет --> K

    K --> TH{Порог релевантности<br/>лучший dense-счёт < 0.55}
    TH -- да --> EMPTY[Выдача = пустая → честный отказ]
    TH -- нет --> HITS[list Chunk + score]
    HITS --> CIT[to_citation n, doc_code, clause,<br/>chunk_id, score, snippet]
    CIT --> EV[sources / followups события]
```

Диаграмма выше — путь чанка от `retrieve` до выдачи. Ключевые решения слоя
(фильтр в каждый `Prefetch`, порог через отдельный dense-запрос, два поля
`snippet`/`text`, payload-ключ `area`) — в `docs/RETRIEVE.md` (разделы 4–7).

---

## 4. Слой данных: Qdrant

Qdrant — векторный индекс, вынесен отдельным compose-слоем
(`deploy/docker-compose.qdrant.yml`). Приложение обращается только к **алиасу**
коллекции (`QDRANT_COLLECTION=aist_chunks`), а не к самой коллекции:

```mermaid
flowchart LR
    subgraph QG [Qdrant :6333]
        subgraph Gen1 [Коллекция aist_chunks_v1]
            D1[dense · 1024d · HNSW m=16]
            B1[bm25 · разреженный · русский язык]
            P1[payload: doc_id, doc_code,<br/>area, section_type, clause,<br/>chunk_id, text, hash, sync_tag]
        end
        subgraph Gen2 [Коллекция aist_chunks_v2]
            D2[dense · другая размерность]
            B2[bm25]
            P2[payload]
        end
        A[алиас aist_chunks<br/>указывает на одну коллекцию]
        A --> Gen1
    end
```

Алиас `aist_chunks` атомарно переключается между поколениями (`<alias>_v1`,
`_v2`…) через `switch_generation`; `upsert_chunks` пишет dense + sparse вместе,
`sweep_document` «подметает» устаревшие точки. BM25 считает сервер Qdrant
(`BM25_OPTIONS = {"language": "russian"}`, `bm25.py:16`); готовность индекса —
`preflight` (`qdrant_search.py:188`). Подробности схемы коллекции, payload и
лингвистики — в `docs/RETRIEVE.md` (раздел 6).

Хранение: Postgres/pgvector (чекпоинтеры графа, история GUI), SQLite в
контейнере Prefect (метаданные оркестрации), ClickHouse — только трассы OpenLIT.

---

## 5. Слой загрузки: индексация корпуса (ingest)

Offline-конвейер превращает сырые документы базы знаний в коллекцию Qdrant.
Полностью — `docs/INGEST.md`; здесь — схема пути одного документа:

```mermaid
flowchart LR
    SRC{Источник}
    SRC -->|files| FS[FilesSource<br/>локальный каталог]
    SRC -->|api| AS[ApiSource<br/>API базы знаний<br/>вход · retry · refresh]

    FS --> RAW[RawDocument]
    AS --> RAW

    RAW --> PARSE{Парсер}
    PARSE -->|html| HP[HtmlParser<br/>html_parse.py]
    PARSE -->|docx| DP[DocxParser<br/>docx_parse.py]

    HP --> DEAN[Deanonymizer<br/>deanon.py]
    DP --> DEAN
    DEAN --> CH[Chunker<br/>chunker.py · атомы → чанки, breadcrumb]
    CH --> CL[AreaClassifier<br/>areas.py · предметные области]
    CL --> EMB[Эмбеддер bge-m3 · vectors]
    EMB --> UP[QdrantStore.upsert_chunks<br/>dense + bm25 + payload]
    UP --> SW[sweep_document]
```

Фазы конвейера (`IngestPipeline.process_document`, pipeline.py:175) — на схеме:
источник → разбор/деанон → чанкинг по атомам → классификация областей → эмбеддинги →
запись + `sweep_document`. Инкремент — хеш документа с версией конвейера
(`incremental.py`); повторный прогон без изменений — 1,76 с на 20 документах.
Коды выхода (`0`/`1`/`2`) и алиас-переключение только при успехе — в `docs/INGEST.md`.

---

## 6. Оркестрация: Prefect-обёртка

Один и тот же конвейер используется и CLI (`scripts/ingest.py`), и Prefect — без
расхождения поведения (`ingest_flow.py:279`, параметр `step` в `pipeline.py:122`):

```mermaid
flowchart TB
    subgraph Trig1 [Ручной запуск из UI · кнопка Run]
        DU[Prefect UI :4200<br/>Deployments → aist-ingest]
    end
    subgraph Trig2 [CLI]
        DC[prefect deployment run ...]
    end
    subgraph Trig3 [Расписание]
        DS[cron из AIST_CRON<br/>напр. каждый день в 02:00]
    end

    Trig1 --> D[Деплоймент aist-ingest<br/>параметры AIST_*]
    Trig2 --> D
    Trig3 --> D

    D --> F[flow ingest_flow]
    F --> T[Родительская задача на документ<br/>документ NNNNN.json]
    T --> P1[загрузка и разбор]
    T --> P2[разбиение на чанки]
    T --> P3[классификация областей]
    T --> P4[эмбеддинги]
    T --> P5[запись в Qdrant]
    F --> A[Артефакт ingest-report · Markdown-сводка]

    P2 -- dry_run: обрыв графа здесь --> F
    P5 --> F
```

Структура задач Prefect (пять фаз как подзадачи, `cache_policy=NO_CACHE` из-за
несериализуемого эмбеддера, артефакт `ingest-report`) и запуск через CLI/server
описаны в `docs/INGEST.md`; параметры `AIST_*`, расписание, отладка и таблица
типовых проблем — в `docs/DEPLOY.md`.

---

## 7. Наблюдаемость: уровень дебага агента — OpenLIT

Трассировка LLM-запросов агента — опциональный слой `obs`
(`deploy/docker-compose.observability.yml`, профиль `obs`, Apache-2.0). В базовый
стек не входит — включается одной переменной `OTLP_ENDPOINT`.

```mermaid
flowchart LR
    A[aegra · граф агента] -->|OTLP HTTP export · /v1/traces| OL[openlit :3000<br/>UI трасс]
    OL -->|INSERT otel_traces| CH[(clickhouse<br/>:8123)]

    subgraph AegraProcess [Процесс aegra]
        TR[setup_tracing · tracer.py:10]
        LI[LangChainInstrumentor]
    end
```

`setup_tracing` (`infrastructure/observability/tracer.py:10`) стартует только при
заданном `OTLP_ENDPOINT` (`service.name=aegra`, экспорт HTTP); инструментарий —
`openinference` `LangChainInstrumentor`: спан пишется на вызов графа/модели,
трассы появляютсяся после первого диалога в GUI. Смотреть в OpenLIT UI `:3000`
(диалог → трасса aegra) или SQL-счётчик в ClickHouse
(`SELECT ServiceName, count() FROM openlit.otel_traces GROUP BY ServiceName`).

Включение, проверка и типовые проблемы — в `docs/DEPLOY.md` («Запуск с
наблюдаемостью», «Устранение неполадок», «Проверка end-to-end»); локальный дебаг
графа и поиска — в `docs/RETRIEVE.md`/`docs/AGENT.md`.

Итого по уровням: интерфейс → граф агента (LangGraph) → поисковый слой (Qdrant,
гибрид dense + BM25 + RRF + порог + реранкер) → генерация (Ollama); рядом —
offline-конвейер индексации, обёрнут в Prefect; сверху — трассировка OpenLIT.
Каждый уровень разобран в своём документе из таблицы в начале.