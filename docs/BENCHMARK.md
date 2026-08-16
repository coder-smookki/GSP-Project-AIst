# Бенчмарк качества

Измерение качества поиска и ответов на золотом наборе, A/B-сравнение
вариантов и история результатов. Этот документ генерируется командой
`uv run eval report` из кэша `eval/results/`.

## Рекомендации

**Текущий лидер:** `LLM=gpt-oss:20b, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks` (граф `base`) — hit@1/3/5 = 0.805/0.927/0.927, MRR = 0.858.

**Значимые эффекты (p < 0.05):**
- `qwen3.8:27b-mlx` → `gemma4:e2b-mlx` (_LLM: qwen3.8:27b-mlx → gemma4:e2b-mlx_): **лучше** — лат., мед.: 0.305→0.294 (p=0.000, улучшение); лат., p95: 0.472→0.326 (p=0.000, улучшение)
- `prod` → `opt` (_мин. релевантность: 0.55 → 0.2; prefetch: 30 → 10_): **лучше** — mrr: 0.776→0.858 (p=0.039, улучшение); лат., мед.: 0.310→0.304 (p=0.000, улучшение); лат., p95: 0.444→0.354 (p=0.000, улучшение)
- `rerank-on` → `rerank-off` (_реранк: вкл → выкл_): **неоднозначно** — hit@1: 0.732→0.561 (p=0.039, ухудшение); mrr: 0.776→0.649 (p=0.028, ухудшение); лат., мед.: 0.297→0.138 (p=0.000, улучшение); лат., p95: 0.345→0.142 (p=0.000, улучшение)

Проверено A/B-сравнений: 6, из них без значимых различий: 3.

## Лидерборд

Каждая строка — уникальная конфигурация (модель, эмбеддер, top_k, реранк), агрегированная по всем прогонам её вариантов. Сортировка — по MRR. Конфигурации разных графов агента не смешиваются: у каждого графа своя секция и свой лидер.

### Граф: `base`

| конфигурация | hit@1 | hit@3 | hit@5 | MRR | not_found | лат., мед. | отказ | кол-во | заметка |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.307 | 1.000 | 1 | максимальное качество |
| LLM=qwen3.8:27b-mlx, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.305 | 1.000 | 1 | — |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=5, реранк=вкл, канд.=10, pref=10, rrf=60, min_rel=0.2, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.304 | 1.000 | 1 | — |
| LLM=qwen3.6:35b-a3b-nvfp4, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.303 | 0.900 | 1 | — |
| LLM=qwen3.5:4b-mlx, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.295 | 0.800 | 1 | — |
| LLM=gemma4:e2b-mlx, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.294 | 0.600 | 1 | — |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=5, реранк=вкл, канд.=10, pref=30, rrf=60, min_rel=0.55, колл.=aist_chunks | 0.732 | 0.805 | 0.854 | 0.776 | 0.146 | 0.303 | 1.000 | 2 | — |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=5, реранк=выкл, канд.=10, pref=30, rrf=60, min_rel=0.55, колл.=aist_chunks | 0.561 | 0.732 | 0.780 | 0.649 | 0.220 | 0.138 | 1.000 | 1 | самый быстрый, реранк выкл |

## Лучшие по графам

Лидер каждого графа агента. Прямое A/B между разными графами не считается: вердикты сравнивают варианты только внутри одного графа.

| граф | лучшая конфигурация | hit@1 | hit@3 | hit@5 | MRR | not_found | кол-во | заметка |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base` | LLM=gpt-oss:20b, emb=bge-m3, top_k=3, реранк=вкл, канд.=10, pref=10, rrf=40, min_rel=0.45, колл.=aist_chunks | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 1 | общий лидер по MRR |

## Как пользоваться

Замер идёт на поднятом стеке (Qdrant, Ollama) с проиндексированным корпусом:
на заглушке поиска команды завершаются с ошибкой, чтобы не выдавать пустые
цифры за результат. Все команды — единый CLI `eval`:

| команда | что делает |
| --- | --- |
| `uv run eval run --with-answers` | качество base на золотом наборе |
| `uv run eval run --variant top6 --settings '{"retrieval_top_k": 6}'` | конкретный вариант |
| `uv run eval ab --variants variants.json --with-answers` | A/B-сравнение вариантов из файла |
| `uv run eval bench --repeats 3` | операционный замер: латенси и нагрузка |
| `uv run eval tune --params '{"retrieval_top_k": [3, 5, 8]}' --objective hit@1` | перебор по сетке параметров |
| `uv run eval report` | обновить этот документ из кэша |
| `uv run eval cache list` | посмотреть, какие результаты сохранены |

Полный отчёт прогона (итоги по каждому вопросу, вердикты) пишется в JSON
флагом `--output`. Результаты кэшируются в `eval/results/` по детерминированному
ключу: повторный прогон без изменений берётся из кэша, `--force` переснимает.

## Метрики

| метрика | что меряет | куда лучше | вопросы |
| --- | --- | --- | --- |
| `hit@1`, `hit@3`, `hit@5` | доля вопросов, где ответ в первых k фрагментах | больше | отвечаемые |
| `mrr` | средняя обратная позиция ответа | больше | отвечаемые |
| `not_found` | доля отвечаемых вопросов без ответа в контексте | меньше | отвечаемые |
| `refusal_rate` | доля вопросов без ответа, где система честно отказала | больше | без ответа |
| `latency_median`, `latency_p95` | время поиска | меньше | все |
| `answer_chars` | средняя длина ответа | — | с ответом |

## Коллекции эмбеддингов

Бенчмарк ходит в коллекцию из настройки `qdrant_collection` (по умолчанию алиас
`aist_chunks`), а эмбеддинги считает моделью из `embedding_model`. Отпечаток
окружения включает обе настройки, поэтому прогоны на разных коллекциях не
смешиваются ни в кэше, ни в лидерборде.

| коллекция | модель эмбеддингов | размерность |
| --- | --- | --- |
| `aist_chunks_v1` | `bge-m3:567m` | 1024 |
| `aist_chunks_v2` | `qwen3-embedding:4b` | 2560 |
| `aist_chunks_v3` | `qwen3-embedding:8b` | 4096 |

Пара «коллекция → модель» обязана быть согласована: индекс построен конкретной
моделью, и запросы должны эмбеддиться той же (размерности различаются). И поиск,
и агент берут модель из одних оверрайдов, поэтому варианту достаточно задать обе
настройки, например файл для `ab`:

    {
      "variants": [
        {"name": "v1-bge",   "settings": {"qdrant_collection": "aist_chunks_v1", "embedding_model": "bge-m3:567m"}},
        {"name": "v2-qwen4b", "settings": {"qdrant_collection": "aist_chunks_v2", "embedding_model": "qwen3-embedding:4b"}},
        {"name": "v3-qwen8b", "settings": {"qdrant_collection": "aist_chunks_v3", "embedding_model": "qwen3-embedding:8b"}}
      ]
    }

    uv run eval ab --variants collections.json --with-answers --bootstraps 5

Алиас `aist_chunks` продолжает указывать на «боевую» коллекцию; конкретные
поколения перебираются оверрайдами без смены алиаса.

## A/B-тест

Варианты сравниваются на одном золотом наборе, поэтому выборки парные.
Категориальные метрики (`hit@k`, `refusal_rate`, `not_found`) — точный тест
МакНемара по расходящимся парам; непрерывные (`mrr`, латенси) — знаково-
ранговый критерий Уилкоксона. Без scipy обе заменяются парным пермутационным
тестом. Вердикт значим при `p < 0.05`.

## Бутстрап-выборки

Один прогон на наборе даёт точечные метрики без представления о разбросе.
Флаг `--bootstraps N` (в `run` и `ab`) ресемплирует пул вопросов **с
возвратом**: вариант N раз прогоняется на детерминированной выборке того же
размера, состав выбирается общим зерном `--seed`, поэтому для всех вариантов
на одной выборке вопросы совпадают. Итоговые метрики — среднее по выборкам,
разброс печатается как `± std`. A/B-сравнение при бутстрепе — по парным
разностям `variant_k − base_k` по выборкам (Уилкоксон, без scipy —
пермутационный тест). Каждая выборка кэшируется отдельно, так что добивание
уже снятых точек ничего не переснимает.

## Тюнинг по сетке

`eval tune` перебирает декартову сетку параметров и сортирует результаты по
целевой метрике — «ручной» шаг поиска оптимума перед точечным A/B:

    uv run eval tune --params '{"retrieval_top_k": [3, 5, 8], "rerank_candidates": [5, 10, 15]}'         --objective hit@1 --direction max --bootstraps 5

Параметры — реальные поля настроек (модель, `retrieval_top_k`,
`rerank_candidates`, `rerank_enabled` и т.д.), имя точки собирается из её
значений. Каждая точка кэшируется, поэтому расширение сетки доганяет только
новые комбинации. `--max-points` (по умолчанию 100) защищает от гигантского
перебора, `--output` пишет полный отчёт с метриками всех точек в JSON.
Итоги сетки включаются в этот документ командой:

    uv run eval report --tune eval/outputs/tune.json

Точки сетки остаются в «Истории результатов», но в лидерборд не попадают —
для них есть раздел «Итоги тюнинга по сетке».

## Итоги тюнинга по сетке

Сетка из 81 точек (граф `base`), цель — `hit@1` (max). Точки кэшированы в `eval/results/`; в лидерборд они не попадают, потому что это инструментальные измерения — сводка здесь.

| параметр | значения |
| --- | --- |
| `retrieval_top_k` | 3, 5, 8 |
| `retrieval_prefetch` | 20, 30, 50 |
| `retrieval_rrf_k` | 40, 60, 80 |
| `retrieval_min_relevance` | 0.3, 0.55, 0.7 |

**Лучшая точка** (по `hit@1`): `retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3`.

| метрика | значение |
| --- | --- |
| hit@1 | 0.805 |
| hit@3 | 0.902 |
| hit@5 | 0.902 |
| mrr | 0.850 |
| not_found | 0.098 |
| latency_median | 0.296 |

### `retrieval_top_k`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 27 | 0.580* | 0.645* | 0.645 | 0.609 | 0.355 | 0.284* |
| 5 | 27 | 0.580 | 0.645 | 0.668 | 0.615 | 0.332 | 0.286 |
| 8 | 27 | 0.580 | 0.645 | 0.670* | 0.615* | 0.328* | 0.292 |

### `retrieval_prefetch`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 27 | 0.602* | 0.667* | 0.678* | 0.634* | 0.322* | 0.286 |
| 30 | 27 | 0.585 | 0.634 | 0.656 | 0.612 | 0.344 | 0.291 |
| 50 | 27 | 0.553 | 0.634 | 0.650 | 0.592 | 0.348 | 0.285* |

### `retrieval_rrf_k`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 27 | 0.580* | 0.645* | 0.662* | 0.613 | 0.337* | 0.287 |
| 60 | 27 | 0.580 | 0.645 | 0.661 | 0.613* | 0.339 | 0.288 |
| 80 | 27 | 0.580 | 0.645 | 0.660 | 0.613 | 0.339 | 0.287* |

### `retrieval_min_relevance`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 27 | 0.772* | 0.870* | 0.894* | 0.822* | 0.105* | 0.287 |
| 0.55 | 27 | 0.724 | 0.821 | 0.846 | 0.773 | 0.154 | 0.287* |
| 0.7 | 27 | 0.244 | 0.244 | 0.244 | 0.244 | 0.756 | 0.288 |

**Направления:** `retrieval_top_k` — не влияет; `retrieval_prefetch` → `20`; `retrieval_rrf_k` — не влияет; `retrieval_min_relevance` → `0.3`.

## Итоги тюнинга по сетке

Сетка из 24 точек (граф `base`), цель — `hit@1` (max). Точки кэшированы в `eval/results/`; в лидерборд они не попадают, потому что это инструментальные измерения — сводка здесь.

| параметр | значения |
| --- | --- |
| `retrieval_min_relevance` | 0.1, 0.2, 0.25, 0.3 |
| `retrieval_prefetch` | 10, 15, 20 |
| `retrieval_top_k` | 3, 5 |

**Лучшая точка** (по `hit@1`): `retrieval_min_relevance=0.1;retrieval_prefetch=10;retrieval_top_k=3`.

| метрика | значение |
| --- | --- |
| hit@1 | 0.805 |
| hit@3 | 0.927 |
| hit@5 | 0.927 |
| mrr | 0.858 |
| not_found | 0.073 |
| latency_median | 0.301 |

### `retrieval_min_relevance`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 6 | 0.805* | 0.911* | 0.919* | 0.854* | 0.081* | 0.287 |
| 0.2 | 6 | 0.805 | 0.911 | 0.919 | 0.854 | 0.081 | 0.283* |
| 0.25 | 6 | 0.805 | 0.911 | 0.919 | 0.854 | 0.081 | 0.284 |
| 0.3 | 6 | 0.805 | 0.911 | 0.919 | 0.854 | 0.081 | 0.283 |

### `retrieval_prefetch`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 8 | 0.805* | 0.927* | 0.927* | 0.858* | 0.073* | 0.286 |
| 15 | 8 | 0.805 | 0.902 | 0.915 | 0.853 | 0.085 | 0.283* |
| 20 | 8 | 0.805 | 0.902 | 0.915 | 0.853 | 0.085 | 0.284 |

### `retrieval_top_k`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 12 | 0.805* | 0.911* | 0.911 | 0.852 | 0.089 | 0.285 |
| 5 | 12 | 0.805 | 0.911 | 0.927* | 0.856* | 0.073* | 0.283* |

**Направления:** `retrieval_min_relevance` — не влияет; `retrieval_prefetch` — не влияет; `retrieval_top_k` — не влияет.

## Итоги тюнинга по сетке

Сетка из 6 точек (граф `base`), цель — `hit@1` (max). Точки кэшированы в `eval/results/`; в лидерборд они не попадают, потому что это инструментальные измерения — сводка здесь.

| параметр | значения |
| --- | --- |
| `retrieval_min_relevance` | 0.3, 0.35, 0.4, 0.45, 0.5, 0.55 |

**Лучшая точка** (по `hit@1`): `retrieval_min_relevance=0.3`.

| метрика | значение |
| --- | --- |
| hit@1 | 0.805 |
| hit@3 | 0.927 |
| hit@5 | 0.927 |
| mrr | 0.858 |
| not_found | 0.073 |
| latency_median | 0.269 |

### `retrieval_min_relevance`

| значение | n | hit@1 | hit@3 | hit@5 | mrr | not_found | latency_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 1 | 0.805* | 0.927* | 0.927* | 0.858* | 0.073* | 0.269* |
| 0.35 | 1 | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.283 |
| 0.4 | 1 | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.282 |
| 0.45 | 1 | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.284 |
| 0.5 | 1 | 0.805 | 0.927 | 0.927 | 0.858 | 0.073 | 0.284 |
| 0.55 | 1 | 0.756 | 0.878 | 0.878 | 0.809 | 0.122 | 0.282 |

**Направления:** `retrieval_min_relevance` → `0.3`.

## История результатов

Каждая строка — один прогон: группа вариантов, снятых на одном наборе,
окружении и конфиге. Сырые строки свёрнуты — обычно нужны только
«Рекомендации» и «Лидерборд» выше. Точки сетки `eval tune` попадают
в историю как есть, но в лидерборд не включаются.

<details>
<summary>сырые прогоны (117)</summary>

| дата | граф | окружение | варианты | метрики | вердикт | кэш |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | gpt-oss:20b | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 1.000<br>latency_median 0.307<br>latency_p95 0.344<br>answer_chars 28.800 | — | [gpt-oss:20b](../eval/results/56e8f5bddf7a61a117804503554f9262.json) |
| 2026-08-16 13:57 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | gemma4:e2b-mlx | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.600<br>latency_median 0.294<br>latency_p95 0.326<br>answer_chars 258.900 | — | [gemma4:e2b-mlx](../eval/results/e25329274b007237013cbd8d5552f839.json) |
| 2026-08-16 13:56 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.5:4b-mlx | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.800<br>latency_median 0.295<br>latency_p95 0.435<br>answer_chars 191.100 | — | [qwen3.5:4b-mlx](../eval/results/c5a5c60232b992d150bb6efb7f94d01f.json) |
| 2026-08-16 13:55 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.6:35b-a3b-nvfp4 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.900<br>latency_median 0.303<br>latency_p95 0.408<br>answer_chars 95.800 | — | [qwen3.6:35b-a3b-nvfp4](../eval/results/deae89a3a36eb23a2e2b308b5cd27c5e.json) |
| 2026-08-16 13:54 | `base` | llm=qwen3.8:27b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 1.000<br>latency_median 0.305<br>latency_p95 0.472<br>answer_chars 29.000 | — | [qwen3.8:27b-mlx](../eval/results/945d6146660a8e05f39f982bed74ec2e.json) |
| 2026-08-16 13:47 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.878<br>hit@5 0.878<br>mrr 0.809<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.282<br>latency_p95 0.312<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.55](../eval/results/4ce2cf5e7c267aea4cc89ec479ebccd0.json) |
| 2026-08-16 13:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.5, coll=aist_chunks | retrieval_min_relevance=0.5 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.200<br>latency_median 0.284<br>latency_p95 0.318<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.5](../eval/results/b218d5124e88bf43861c80dbc98ce65e.json) |
| 2026-08-16 13:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | retrieval_min_relevance=0.45 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.100<br>latency_median 0.284<br>latency_p95 0.320<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.45](../eval/results/e0fc9be87434a16145e123a838f428fa.json) |
| 2026-08-16 13:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.4, coll=aist_chunks | retrieval_min_relevance=0.4 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.282<br>latency_p95 0.317<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.4](../eval/results/9629c2f8863182d250e31c23f97b38a8.json) |
| 2026-08-16 13:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.35, coll=aist_chunks | retrieval_min_relevance=0.35 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.359<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.35](../eval/results/69d497839eaf37105453c712a7b55d07.json) |
| 2026-08-16 13:45 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.269<br>latency_p95 0.335<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.3](../eval/results/32fd86ae21f1eb0e358a9d5437566532.json) |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | opt | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 1.000<br>latency_median 0.304<br>latency_p95 0.354<br>answer_chars 28.900 | — | [opt](../eval/results/de09a8547b184f2c452e5e6d8bd1f51a.json) |
| 2026-08-16 13:11 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_min_relevance=0.3;retrieval_prefetch=15;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.282<br>latency_p95 0.312<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.3;retrieval_prefetch=15;retrieval_top_k=5](../eval/results/807d04e63bc7b39eeb3a32741a4466aa.json) |
| 2026-08-16 13:11 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_min_relevance=0.3;retrieval_prefetch=15;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.317<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.3;retrieval_prefetch=15;retrieval_top_k=3](../eval/results/49c1de71ab249ce7f1547d752497ca8e.json) |
| 2026-08-16 13:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_min_relevance=0.3;retrieval_prefetch=10;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.285<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.3;retrieval_prefetch=10;retrieval_top_k=5](../eval/results/ed4524f4764ade6bc14bb5ef2f1696be.json) |
| 2026-08-16 13:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_min_relevance=0.3;retrieval_prefetch=10;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.308<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.3;retrieval_prefetch=10;retrieval_top_k=3](../eval/results/86e0d015e89e29f6543673ed7f1a29c9.json) |
| 2026-08-16 13:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.25, coll=aist_chunks | retrieval_min_relevance=0.25;retrieval_prefetch=20;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.285<br>latency_p95 0.313<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.25;retrieval_prefetch=20;retrieval_top_k=5](../eval/results/a4024f536377f467c00fe25f00108e4f.json) |
| 2026-08-16 13:09 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.25, coll=aist_chunks | retrieval_min_relevance=0.25;retrieval_prefetch=20;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.313<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.25;retrieval_prefetch=20;retrieval_top_k=3](../eval/results/d49682f42de28c42d7dd415303f363f3.json) |
| 2026-08-16 13:09 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.25, coll=aist_chunks | retrieval_min_relevance=0.25;retrieval_prefetch=15;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.25;retrieval_prefetch=15;retrieval_top_k=5](../eval/results/eab1ae628be1a4bf2c5f4b09d43607fb.json) |
| 2026-08-16 13:09 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.25, coll=aist_chunks | retrieval_min_relevance=0.25;retrieval_prefetch=15;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.318<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.25;retrieval_prefetch=15;retrieval_top_k=3](../eval/results/e783042cb405b3c96481067179cfc627.json) |
| 2026-08-16 13:08 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.25, coll=aist_chunks | retrieval_min_relevance=0.25;retrieval_prefetch=10;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.309<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.25;retrieval_prefetch=10;retrieval_top_k=5](../eval/results/88959d043effce3c4a006b76f807e83f.json) |
| 2026-08-16 13:08 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.25, coll=aist_chunks | retrieval_min_relevance=0.25;retrieval_prefetch=10;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.285<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.25;retrieval_prefetch=10;retrieval_top_k=3](../eval/results/578d790132defec29abab9cecbae8042.json) |
| 2026-08-16 13:08 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | retrieval_min_relevance=0.2;retrieval_prefetch=20;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.311<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.2;retrieval_prefetch=20;retrieval_top_k=5](../eval/results/1d8f5055de8a678acf229449334bb6fe.json) |
| 2026-08-16 13:08 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | retrieval_min_relevance=0.2;retrieval_prefetch=20;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.2;retrieval_prefetch=20;retrieval_top_k=3](../eval/results/e005fd646b301cbce0035d68a538e5a6.json) |
| 2026-08-16 13:07 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | retrieval_min_relevance=0.2;retrieval_prefetch=15;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.281<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.2;retrieval_prefetch=15;retrieval_top_k=5](../eval/results/9cfaa39f64acb3e91a32997ce39562b9.json) |
| 2026-08-16 13:07 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | retrieval_min_relevance=0.2;retrieval_prefetch=15;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.308<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.2;retrieval_prefetch=15;retrieval_top_k=3](../eval/results/f682f08ac9885fcb652d5c96094cd247.json) |
| 2026-08-16 13:07 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | retrieval_min_relevance=0.2;retrieval_prefetch=10;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.282<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.2;retrieval_prefetch=10;retrieval_top_k=5](../eval/results/bd8303805190579d0a1a28018dd46ce0.json) |
| 2026-08-16 13:06 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | retrieval_min_relevance=0.2;retrieval_prefetch=10;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.285<br>latency_p95 0.313<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.2;retrieval_prefetch=10;retrieval_top_k=3](../eval/results/a915f9a57e2870bf559f5a1edb66c35f.json) |
| 2026-08-16 13:06 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.1, coll=aist_chunks | retrieval_min_relevance=0.1;retrieval_prefetch=20;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.318<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.1;retrieval_prefetch=20;retrieval_top_k=5](../eval/results/9f3812f1ca39283973d69f1499b05098.json) |
| 2026-08-16 13:06 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.1, coll=aist_chunks | retrieval_min_relevance=0.1;retrieval_prefetch=20;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.286<br>latency_p95 0.317<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.1;retrieval_prefetch=20;retrieval_top_k=3](../eval/results/fb825e990c6e139be90ae26b72bccf12.json) |
| 2026-08-16 13:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.1, coll=aist_chunks | retrieval_min_relevance=0.1;retrieval_prefetch=15;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.1;retrieval_prefetch=15;retrieval_top_k=5](../eval/results/fcb8d5af825d382aad92af5c143fd1b7.json) |
| 2026-08-16 13:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=15, rrf=60, rerank=вкл, cand=10, min_rel=0.1, coll=aist_chunks | retrieval_min_relevance=0.1;retrieval_prefetch=15;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.313<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.1;retrieval_prefetch=15;retrieval_top_k=3](../eval/results/99b1ab47d1e89c046b457fb9ab590dbc.json) |
| 2026-08-16 13:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.1, coll=aist_chunks | retrieval_min_relevance=0.1;retrieval_prefetch=10;retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.347<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.1;retrieval_prefetch=10;retrieval_top_k=5](../eval/results/6c99b93f2966cbb21716232ae17e0139.json) |
| 2026-08-16 13:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.1, coll=aist_chunks | retrieval_min_relevance=0.1;retrieval_prefetch=10;retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.927<br>hit@5 0.927<br>mrr 0.858<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.301<br>latency_p95 0.326<br>answer_chars 0.000 | — | [retrieval_min_relevance=0.1;retrieval_prefetch=10;retrieval_top_k=3](../eval/results/2f186ce6ef79c135257e056e3db425ac.json) |
| 2026-08-16 12:07 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.293<br>latency_p95 0.350<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/616eef3b971267248fb15c5aea4d6b37.json) |
| 2026-08-16 12:06 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.747<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.288<br>latency_p95 0.338<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/b42b9a52e312786441bfac760ffaa371.json) |
| 2026-08-16 12:06 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.795<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.283<br>latency_p95 0.327<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/c01a9a455cb1ae20c4a20d1ce9b5cfe1.json) |
| 2026-08-16 12:06 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.283<br>latency_p95 0.322<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/fdc0c8594a55813d80d93fcd55f89ac3.json) |
| 2026-08-16 12:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.747<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.287<br>latency_p95 0.342<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/dfdb17c59a88db9e2799f023c564c8fd.json) |
| 2026-08-16 12:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.796<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.287<br>latency_p95 0.328<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/a8656fa320545fc01b57e455e8552eb2.json) |
| 2026-08-16 12:05 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.286<br>latency_p95 0.331<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/fb5f22c57455ef2bf730d4fabdc7a577.json) |
| 2026-08-16 12:04 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.829<br>mrr 0.746<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.288<br>latency_p95 0.331<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/76f5bedf91dc27fe549741503e53e8a6.json) |
| 2026-08-16 12:04 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.791<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.286<br>latency_p95 0.333<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/995d9af38466e7197653d6621fc6b53e.json) |
| 2026-08-16 12:04 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.288<br>latency_p95 0.321<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/8e42385be380328955c783cc6ca2d960.json) |
| 2026-08-16 12:03 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.288<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/fa94a5e0ae6f35609c1dcd1a63231c2b.json) |
| 2026-08-16 12:03 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.825<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.290<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/ea67ec5f23b699738fe66d77c31744c8.json) |
| 2026-08-16 12:03 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.288<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/8099731a0f23a98421baa211a465b284.json) |
| 2026-08-16 12:02 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.285<br>latency_p95 0.321<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/cce1d862697bd33e5ad4f848fe1da3ac.json) |
| 2026-08-16 12:02 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.825<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.303<br>latency_p95 0.334<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/131e87fc373a122d1e3e63ab1dc3c5ca.json) |
| 2026-08-16 12:02 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.307<br>latency_p95 0.335<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/b60c11712ced3803737406b359dd5d6f.json) |
| 2026-08-16 12:01 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.307<br>latency_p95 0.336<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/58b29fac59638a4e53574f374d83207a.json) |
| 2026-08-16 12:01 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.825<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.307<br>latency_p95 0.336<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/b5a3bac16416672e537d1fd2de8a7ead.json) |
| 2026-08-16 12:01 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.302<br>latency_p95 0.335<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/3d083e64fc7257510c137d5062f846bf.json) |
| 2026-08-16 12:01 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.807<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.304<br>latency_p95 0.329<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/1b8018f01d6d40e3d3d9106b4078186f.json) |
| 2026-08-16 12:00 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.302<br>latency_p95 0.332<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/e3ee722171f7d9d31c57b6a562bb0d40.json) |
| 2026-08-16 12:00 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.302<br>latency_p95 0.336<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/a7cf75958deb71975140e7e00f230fda.json) |
| 2026-08-16 12:00 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.807<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.289<br>latency_p95 0.327<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/9ec16582287cf77fe86558d4e3f45c53.json) |
| 2026-08-16 11:59 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.279<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/cb95461a10ce93cecc33cba367145f8e.json) |
| 2026-08-16 11:59 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.285<br>latency_p95 0.317<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/1931f5d72ce9aa3b3a92cef9e5e71b35.json) |
| 2026-08-16 11:59 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.807<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.284<br>latency_p95 0.315<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/c391a65ee4695fa25f86031bb0b84b13.json) |
| 2026-08-16 11:58 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_top_k=8;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/a5c343ada776250e89e218cc666cafbb.json) |
| 2026-08-16 11:58 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.287<br>latency_p95 0.329<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/72a9ccc483838a7f7a0dedd96f76f5a7.json) |
| 2026-08-16 11:58 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.829<br>mrr 0.742<br>not_found 0.171<br>refusal_rate 0.500<br>latency_median 0.283<br>latency_p95 0.341<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/b972f00a1f2386464cf284cd9a7cf978.json) |
| 2026-08-16 11:57 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.791<br>not_found 0.122<br>refusal_rate 0.000<br>latency_median 0.287<br>latency_p95 0.330<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/3d6020b542c066c91b21215e9df76045.json) |
| 2026-08-16 11:57 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.287<br>latency_p95 0.330<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/5040d50576006b071c4df68cf1bd6496.json) |
| 2026-08-16 11:57 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.829<br>mrr 0.742<br>not_found 0.171<br>refusal_rate 0.500<br>latency_median 0.286<br>latency_p95 0.344<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/d744028d0cebae06a246265ca1e93f7c.json) |
| 2026-08-16 11:56 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.791<br>not_found 0.122<br>refusal_rate 0.000<br>latency_median 0.286<br>latency_p95 0.343<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/9ad5f03b98d5b7a3ef46e38fb7b55aef.json) |
| 2026-08-16 11:56 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.287<br>latency_p95 0.330<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/9aeb73cd90058b111a8c15df4d5a6121.json) |
| 2026-08-16 11:56 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.747<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.286<br>latency_p95 0.331<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/009fcb9937c5f8093cbfb12cad0985a8.json) |
| 2026-08-16 11:55 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.796<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.282<br>latency_p95 0.337<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/74688e22a3a847cc73a9fffa7a0ee5b5.json) |
| 2026-08-16 11:55 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.288<br>latency_p95 0.318<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/c17bb3eed04e7ff559222ca48996482f.json) |
| 2026-08-16 11:55 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.286<br>latency_p95 0.311<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/480367676e38d605475b9bb703fa32dd.json) |
| 2026-08-16 11:54 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.825<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.288<br>latency_p95 0.328<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/7154781fc27b0f4135eb3bb136761ca3.json) |
| 2026-08-16 11:54 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.306<br>latency_p95 0.340<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/480cb4aa39cd9800b3f03e059e33ac42.json) |
| 2026-08-16 11:54 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.301<br>latency_p95 0.334<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/57ce701eb3374cb2901a9395aa1712fe.json) |
| 2026-08-16 11:53 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.825<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.301<br>latency_p95 0.332<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/330ce22ee0cf6a20cb927267d60a0653.json) |
| 2026-08-16 11:53 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.284<br>latency_p95 0.317<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/9da9e0e4c2d024e168ad507e9590e2fe.json) |
| 2026-08-16 11:53 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.285<br>latency_p95 0.312<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/d839b65a8d1c13cf3aa426a731391d5a.json) |
| 2026-08-16 11:53 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.902<br>mrr 0.825<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.285<br>latency_p95 0.318<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/2ce743715eb4da5d6488e1d532f0c0fa.json) |
| 2026-08-16 11:52 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.281<br>latency_p95 0.309<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/99537e9274b483fa5aa18d4611d0738d.json) |
| 2026-08-16 11:52 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.807<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.281<br>latency_p95 0.310<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/738559003981a2203181553039edab8e.json) |
| 2026-08-16 11:52 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.281<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/998953e9dbd8447b3954d10d5c38cbc2.json) |
| 2026-08-16 11:51 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.281<br>latency_p95 0.308<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/76ebbc5472cf30cd44f23b2eb4f2f953.json) |
| 2026-08-16 11:51 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.807<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.281<br>latency_p95 0.310<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/a6b244a5eab505b6aca943a0c7e5ac4f.json) |
| 2026-08-16 11:51 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3, retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5 | hit@1 retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.805 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.805<br>hit@3 retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.902 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.902<br>hit@5 retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.927 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.927<br>mrr retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.856 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.856<br>not_found retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.073 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.073<br>refusal_rate retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.000 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.000<br>latency_median retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.282 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.284<br>latency_p95 retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.315 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.309<br>answer_chars retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.000 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5=0.000 | retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5: без значимых различий | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/c8c1d3794c2340f20dd9938a73c04a09.json)<br>[retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=5](../eval/results/15ce54b40a17b7a016b7cb8fae089d66.json) |
| 2026-08-16 11:50 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.280<br>latency_p95 0.315<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/177ed889172cb7bb16be7332e120358f.json) |
| 2026-08-16 11:50 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.878<br>mrr 0.807<br>not_found 0.122<br>refusal_rate 0.500<br>latency_median 0.280<br>latency_p95 0.311<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/21eb98d2a8172180c88c2048c6b54637.json) |
| 2026-08-16 11:50 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.856<br>not_found 0.073<br>refusal_rate 0.000<br>latency_median 0.280<br>latency_p95 0.313<br>answer_chars 0.000 | — | [retrieval_top_k=5;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/b540e12d3288f012e0da4cdd3d2073a1.json) |
| 2026-08-16 11:49 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.284<br>latency_p95 0.328<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/1f3b264f0674c13cc267c11272f087ea.json) |
| 2026-08-16 11:49 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.736<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.279<br>latency_p95 0.334<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/7c06e50ecac9991ed6ee52ffca30e1e1.json) |
| 2026-08-16 11:49 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.785<br>not_found 0.146<br>refusal_rate 0.000<br>latency_median 0.281<br>latency_p95 0.329<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/3c53b5f003d1d73ae2398cc4074ae89f.json) |
| 2026-08-16 11:48 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.281<br>latency_p95 0.329<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/450d743794ea3714e75011fbe62e5a6a.json) |
| 2026-08-16 11:48 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.736<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.283<br>latency_p95 0.337<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/ce4e8c6693eaf05a4d6051352c971966.json) |
| 2026-08-16 11:48 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.785<br>not_found 0.146<br>refusal_rate 0.000<br>latency_median 0.286<br>latency_p95 0.342<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/1cb5f2a207594f589bc9bceea46e7760.json) |
| 2026-08-16 11:48 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.283<br>latency_p95 0.329<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/cb67cedbc12c7a2cc3d44d8e3697ec15.json) |
| 2026-08-16 11:47 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.683<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.732<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.284<br>latency_p95 0.328<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/83bcf1feb415298293e6c425d0ba0e15.json) |
| 2026-08-16 11:47 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=50, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.732<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.780<br>not_found 0.146<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.345<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=50;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/a20ffccf5ccbc4859c9d9ed6799f82b0.json) |
| 2026-08-16 11:47 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.286<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/c411e0cb51656483ed564d1d1c9ced07.json) |
| 2026-08-16 11:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.764<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.286<br>latency_p95 0.312<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/770e5d54034c2a403ab45fe5542b8e00.json) |
| 2026-08-16 11:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.813<br>not_found 0.146<br>refusal_rate 0.000<br>latency_median 0.286<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/b869c1ba7bba9de4cb58e34e1d3ca65e.json) |
| 2026-08-16 11:46 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.287<br>latency_p95 0.318<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/0242957c2a7373344345aa8ca631c431.json) |
| 2026-08-16 11:45 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.764<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.286<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/9cef596130e1b87974ae23c48d3183d8.json) |
| 2026-08-16 11:45 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.813<br>not_found 0.146<br>refusal_rate 0.000<br>latency_median 0.284<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/fbc639535993a1c1df82668f5b9e44ac.json) |
| 2026-08-16 11:45 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.286<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/eb2e968103ed1e506b5ce1af7c640ed0.json) |
| 2026-08-16 11:44 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.764<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.287<br>latency_p95 0.316<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/0249582ce1a42783a5ff5de09b0fac42.json) |
| 2026-08-16 11:44 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.780<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.813<br>not_found 0.146<br>refusal_rate 0.000<br>latency_median 0.289<br>latency_p95 0.321<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=30;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/d8f25c180f7c206fb2f8090659a663a7.json) |
| 2026-08-16 11:44 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.283<br>latency_p95 0.317<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.7](../eval/results/864672f02f5770727a89e864f8d8898f.json) |
| 2026-08-16 11:43 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.801<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.282<br>latency_p95 0.311<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.55](../eval/results/3dc5bdb24938b1365034aa67114bac3c.json) |
| 2026-08-16 11:43 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=80, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.278<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=80;retrieval_min_relevance=0.3](../eval/results/3d8f9f27b3a6b879f2a6790f06d1a84b.json) |
| 2026-08-16 11:43 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.284<br>latency_p95 0.315<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.7](../eval/results/33ebe44adb814b2653393ebf7d374444.json) |
| 2026-08-16 11:42 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.801<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.286<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.55](../eval/results/f67eaf356149a036f288aee87b6f04e6.json) |
| 2026-08-16 11:42 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=60, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3, retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3 | hit@1 retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.805 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.805<br>hit@3 retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.902 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.902<br>hit@5 retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.902 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.902<br>mrr retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.850 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.850<br>not_found retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.098 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.098<br>refusal_rate retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.000 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.000<br>latency_median retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.281 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.283<br>latency_p95 retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.314 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.311<br>answer_chars retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3=0.000 / retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3=0.000 | retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3 (лучше: latency_p95; хуже: latency_median) | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=60;retrieval_min_relevance=0.3](../eval/results/3dc935d70328260d0f574a3877025a50.json)<br>[retrieval_min_relevance=0.3;retrieval_prefetch=20;retrieval_top_k=3](../eval/results/6a499347ee824c3b69a464ce41ff7367.json) |
| 2026-08-16 11:42 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.7, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.7 | hit@1 0.244<br>hit@3 0.244<br>hit@5 0.244<br>mrr 0.244<br>not_found 0.756<br>refusal_rate 1.000<br>latency_median 0.283<br>latency_p95 0.314<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.7](../eval/results/7aa44dcd4efe46a396281a3c1caa2fdf.json) |
| 2026-08-16 11:42 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.55 | hit@1 0.756<br>hit@3 0.854<br>hit@5 0.854<br>mrr 0.801<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.283<br>latency_p95 0.320<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.55](../eval/results/d09b346d911f35e50a842088e15802ec.json) |
| 2026-08-16 11:41 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=20, rrf=40, rerank=вкл, cand=10, min_rel=0.3, coll=aist_chunks | retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.850<br>not_found 0.098<br>refusal_rate 0.000<br>latency_median 0.296<br>latency_p95 0.334<br>answer_chars 0.000 | — | [retrieval_top_k=3;retrieval_prefetch=20;retrieval_rrf_k=40;retrieval_min_relevance=0.3](../eval/results/d08384125806a79ab586340012d9dfde.json) |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-off | hit@1 0.561<br>hit@3 0.732<br>hit@5 0.780<br>mrr 0.649<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.138<br>latency_p95 0.142<br>answer_chars 28.600 | — | [rerank-off](../eval/results/fe01fb92d638d0e0eb77737a113689b0.json) |
| 2026-08-16 11:35 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on, prod | hit@1 rerank-on=0.732 / prod=0.732<br>hit@3 rerank-on=0.805 / prod=0.805<br>hit@5 rerank-on=0.854 / prod=0.854<br>mrr rerank-on=0.776 / prod=0.776<br>not_found rerank-on=0.146 / prod=0.146<br>refusal_rate rerank-on=1.000 / prod=1.000<br>latency_median rerank-on=0.297 / prod=0.310<br>latency_p95 rerank-on=0.345 / prod=0.444<br>answer_chars rerank-on=28.600 / prod=28.700 | prod (хуже: latency_median, latency_p95) | [rerank-on](../eval/results/05e2f8f2b087eff1974532ed2d670f4c.json)<br>[prod](../eval/results/2fd0522a070901cb66234faa703a6c8e.json) |

</details>


## A/B сравнения (сводка)

По одной строке на пару вариантов. Показаны только значимые отличия (p < 0.05); остальные метрики не выводятся, чтобы не засорять таблицу константами.

| дата | граф | тест (base → variant) | что изменилось | итог | значимые отличия (p < 0.05) |
| --- | --- | --- | --- | --- | --- |
| 2026-08-16 14:10 | `base` | `qwen3.8:27b-mlx` → `gpt-oss:20b` | LLM: qwen3.8:27b-mlx → gpt-oss:20b | **паритет** | — |
| 2026-08-16 14:10 | `base` | `qwen3.8:27b-mlx` → `gemma4:e2b-mlx` | LLM: qwen3.8:27b-mlx → gemma4:e2b-mlx | **лучше** | лат., мед.: 0.305→0.294 (p=0.000, улучшение); лат., p95: 0.472→0.326 (p=0.000, улучшение) |
| 2026-08-16 14:10 | `base` | `qwen3.8:27b-mlx` → `qwen3.5:4b-mlx` | LLM: qwen3.8:27b-mlx → qwen3.5:4b-mlx | **паритет** | — |
| 2026-08-16 14:10 | `base` | `qwen3.8:27b-mlx` → `qwen3.6:35b-a3b-nvfp4` | LLM: qwen3.8:27b-mlx → qwen3.6:35b-a3b-nvfp4 | **паритет** | — |
| 2026-08-16 13:40 | `base` | `prod` → `opt` | мин. релевантность: 0.55 → 0.2; prefetch: 30 → 10 | **лучше** | mrr: 0.776→0.858 (p=0.039, улучшение); лат., мед.: 0.310→0.304 (p=0.000, улучшение); лат., p95: 0.444→0.354 (p=0.000, улучшение) |
| 2026-08-16 11:36 | `base` | `rerank-on` → `rerank-off` | реранк: вкл → выкл | **неоднозначно** | hit@1: 0.732→0.561 (p=0.039, ухудшение); mrr: 0.776→0.649 (p=0.028, ухудшение); лат., мед.: 0.297→0.138 (p=0.000, улучшение); лат., p95: 0.345→0.142 (p=0.000, улучшение) |

<details>
<summary>все сравнения по метрикам (54)</summary>

«Δ» — разница variant - base. Вердикт: «лучше» — значимое улучшение (p < 0.05), «хуже» — ухудшение, «без разницы» — нет значимых отличий.

| дата | граф | окружение | base | variant | метрика | base | variant | Δ | p-value | вердикт | тест |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | hit@1 | 0.805 | 0.805 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | hit@3 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | hit@5 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | mrr | 0.858 | 0.858 | +0.000 | 1.000 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | not_found | 0.073 | 0.073 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | refusal_rate | 1.000 | 1.000 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | latency_median | 0.305 | 0.307 | +0.002 | 0.660 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | latency_p95 | 0.472 | 0.344 | -0.128 | 0.660 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gpt-oss:20b | answer_chars | 29.000 | 28.800 | -0.200 | 0.157 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | hit@1 | 0.805 | 0.805 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | hit@3 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | hit@5 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | mrr | 0.858 | 0.858 | +0.000 | 1.000 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | not_found | 0.073 | 0.073 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | refusal_rate | 1.000 | 0.600 | -0.400 | 0.125 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | latency_median | 0.305 | 0.294 | -0.011 | 0.000 | лучше | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | latency_p95 | 0.472 | 0.326 | -0.146 | 0.000 | лучше | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=gemma4:e2b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | gemma4:e2b-mlx | answer_chars | 29.000 | 258.900 | +229.900 | 0.018 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | hit@1 | 0.805 | 0.805 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | hit@3 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | hit@5 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | mrr | 0.858 | 0.858 | +0.000 | 1.000 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | not_found | 0.073 | 0.073 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | refusal_rate | 1.000 | 0.800 | -0.200 | 0.500 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | latency_median | 0.305 | 0.295 | -0.010 | 0.070 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | latency_p95 | 0.472 | 0.435 | -0.037 | 0.070 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.5:4b-mlx, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.5:4b-mlx | answer_chars | 29.000 | 191.100 | +162.100 | 0.109 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | hit@1 | 0.805 | 0.805 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | hit@3 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | hit@5 | 0.927 | 0.927 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | mrr | 0.858 | 0.858 | +0.000 | 1.000 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | not_found | 0.073 | 0.073 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | refusal_rate | 1.000 | 0.900 | -0.100 | 1.000 | без разницы | mcnemar |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | latency_median | 0.305 | 0.303 | -0.002 | 0.277 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | latency_p95 | 0.472 | 0.408 | -0.064 | 0.277 | без разницы | wilcoxon |
| 2026-08-16 14:10 | `base` | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=3, prefetch=10, rrf=40, rerank=вкл, cand=10, min_rel=0.45, coll=aist_chunks | qwen3.8:27b-mlx | qwen3.6:35b-a3b-nvfp4 | answer_chars | 29.000 | 95.800 | +66.800 | 1.000 | без разницы | wilcoxon |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | hit@1 | 0.732 | 0.805 | +0.073 | 0.250 | без разницы | mcnemar |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | hit@3 | 0.805 | 0.927 | +0.122 | 0.062 | без разницы | mcnemar |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | hit@5 | 0.854 | 0.927 | +0.073 | 0.250 | без разницы | mcnemar |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | mrr | 0.776 | 0.858 | +0.081 | 0.039 | лучше | wilcoxon |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | not_found | 0.146 | 0.073 | -0.073 | 0.250 | без разницы | mcnemar |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | refusal_rate | 1.000 | 1.000 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | latency_median | 0.310 | 0.304 | -0.006 | 0.000 | лучше | wilcoxon |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | latency_p95 | 0.444 | 0.354 | -0.089 | 0.000 | лучше | wilcoxon |
| 2026-08-16 13:40 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=10, rrf=60, rerank=вкл, cand=10, min_rel=0.2, coll=aist_chunks | prod | opt | answer_chars | 28.700 | 28.900 | +0.200 | 0.157 | без разницы | wilcoxon |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | hit@1 | 0.732 | 0.561 | -0.171 | 0.039 | хуже | mcnemar |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | hit@3 | 0.805 | 0.732 | -0.073 | 0.375 | без разницы | mcnemar |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | hit@5 | 0.854 | 0.780 | -0.073 | 0.250 | без разницы | mcnemar |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | mrr | 0.776 | 0.649 | -0.127 | 0.028 | хуже | wilcoxon |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | not_found | 0.146 | 0.220 | +0.073 | 0.250 | без разницы | mcnemar |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | refusal_rate | 1.000 | 1.000 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | latency_median | 0.297 | 0.138 | -0.159 | 0.000 | лучше | wilcoxon |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | latency_p95 | 0.345 | 0.142 | -0.202 | 0.000 | лучше | wilcoxon |
| 2026-08-16 11:36 | `base` | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=выкл, cand=10, min_rel=0.55, coll=aist_chunks | rerank-on | rerank-off | answer_chars | 28.600 | 28.600 | +0.000 | 1.000 | без разницы | wilcoxon |

</details>
