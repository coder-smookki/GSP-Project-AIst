# Бенчмарк качества

Измерение качества поиска и ответов на золотом наборе, A/B-сравнение
вариантов и история результатов. Этот документ генерируется командой
`uv run eval report` из кэша `eval/results/`.

## Рекомендации

**Текущий лидер:** `LLM=gpt-oss:20b, emb=qwen3-emb, top_k=5, реранк=вкл, канд.=10, колл.=aist_chunks_v2` — hit@1 = 1.000, MRR = 1.000.

**Значимых A/B-эффектов (p < 0.05) пока нет.**

## Лидерборд

Каждая строка — уникальная конфигурация (модель, эмбеддер, top_k, реранк), агрегированная по всем прогонам её вариантов. Сортировка — по MRR.

| конфигурация | hit@1 | MRR | not_found | лат., мед. | кол-во | заметка |
| --- | --- | --- | --- | --- | --- | --- |
| LLM=gpt-oss:20b, emb=qwen3-emb, top_k=5, реранк=вкл, канд.=10, колл.=aist_chunks_v2 | 1.000 | 1.000 | 0.000 | 0.372 | 1 | максимальное качество |
| LLM=gpt-oss:20b, emb=qwen3-emb, top_k=5, реранк=вкл, канд.=10 | 0.667 | 0.833 | 0.000 | 0.350 | 2 | — |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=8, реранк=вкл, канд.=10 | 0.744 | 0.787 | 0.159 | 0.281 | 2 | самый быстрый |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=5, реранк=вкл, канд.=10 | 0.738 | 0.782 | 0.152 | 0.290 | 4 | — |
| LLM=gpt-oss:20b, emb=bge-m3, top_k=3, реранк=вкл, канд.=10 | 0.740 | 0.775 | 0.187 | 0.290 | 3 | — |

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

## История результатов

Каждая строка — один прогон: группа вариантов, снятых на одном наборе,
окружении и конфиге. Сырые строки свёрнуты — обычно нужны только
«Рекомендации» и «Лидерборд» выше.

<details>
<summary>сырые прогоны (12)</summary>

| дата | окружение | варианты | метрики | вердикт | кэш |
| --- | --- | --- | --- | --- | --- |
| 2026-08-16 10:56 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55, coll=aist_chunks_v2 | v2 | hit@1 1.000<br>hit@3 1.000<br>hit@5 1.000<br>mrr 1.000<br>not_found 0.000<br>refusal_rate 1.000<br>latency_median 0.372<br>latency_p95 7.966<br>answer_chars 0.000 | — | [v2](../eval/results/2ad4c61ea9b4051f5711b1ec7ca3459a.json) |
| 2026-08-16 10:52 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | v2 | hit@1 0.667<br>hit@3 1.000<br>hit@5 1.000<br>mrr 0.833<br>not_found 0.000<br>refusal_rate 1.000<br>latency_median 0.361<br>latency_p95 5.503<br>answer_chars 28.000 | — | [v2](../eval/results/cebfed4f77fb5b979745ca2c3acc08d1.json) |
| 2026-08-16 10:51 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | v2 | hit@1 0.667<br>hit@3 1.000<br>hit@5 1.000<br>mrr 0.833<br>not_found 0.000<br>refusal_rate 1.000<br>latency_median 0.340<br>latency_p95 6.402<br>answer_chars 0.000 | — | [v2](../eval/results/0a1100c264bb2c12151b2f4b5c35034f.json) |
| 2026-08-16 08:42 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=8 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.860<br>not_found 0.073<br>refusal_rate 0.300<br>latency_median 0.283<br>latency_p95 0.327<br>answer_chars 0.000 | — | [retrieval_top_k=8](../eval/results/5283b34ba08b44e55fce4c1b601f9540.json) |
| 2026-08-16 08:42 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=8, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=8 | hit@1 0.683<br>hit@3 0.732<br>hit@5 0.756<br>mrr 0.713<br>not_found 0.244<br>refusal_rate 0.700<br>latency_median 0.278<br>latency_p95 0.302<br>answer_chars 0.000 | — | [retrieval_top_k=8](../eval/results/20656a62a7a87d7dc6b2618b697750a0.json) |
| 2026-08-16 08:42 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=5 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.927<br>mrr 0.860<br>not_found 0.073<br>refusal_rate 0.300<br>latency_median 0.284<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_top_k=5](../eval/results/d043acfa8f6f40e96c04933f4909cc07.json) |
| 2026-08-16 08:41 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=5 | hit@1 0.683<br>hit@3 0.732<br>hit@5 0.756<br>mrr 0.713<br>not_found 0.244<br>refusal_rate 0.700<br>latency_median 0.278<br>latency_p95 0.302<br>answer_chars 0.000 | — | [retrieval_top_k=5](../eval/results/58c014c02c0f11f9c66f1e0bb16621fe.json) |
| 2026-08-16 08:41 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=3 | hit@1 0.805<br>hit@3 0.902<br>hit@5 0.902<br>mrr 0.854<br>not_found 0.098<br>refusal_rate 0.300<br>latency_median 0.286<br>latency_p95 0.334<br>answer_chars 0.000 | — | [retrieval_top_k=3](../eval/results/4f7fb4aea477ebc6cdd69ee1aa74e606.json) |
| 2026-08-16 08:41 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=3 | hit@1 0.683<br>hit@3 0.732<br>hit@5 0.732<br>mrr 0.707<br>not_found 0.268<br>refusal_rate 0.700<br>latency_median 0.290<br>latency_p95 0.341<br>answer_chars 0.000 | — | [retrieval_top_k=3](../eval/results/1ca21a9dd7b917574e4954fcd7cf3ca2.json) |
| 2026-08-16 08:40 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=5 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.500<br>latency_median 0.285<br>latency_p95 0.319<br>answer_chars 0.000 | — | [retrieval_top_k=5](../eval/results/c044e8e0f6615304201a43ea9740943e.json) |
| 2026-08-16 08:40 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=3, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | retrieval_top_k=3 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.805<br>mrr 0.764<br>not_found 0.195<br>refusal_rate 0.500<br>latency_median 0.293<br>latency_p95 0.322<br>answer_chars 0.000 | — | [retrieval_top_k=3](../eval/results/8b36b5b6f238841f7e6b39097e554555.json) |
| 2026-08-16 08:12 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, prefetch=30, rrf=60, rerank=вкл, cand=10, min_rel=0.55 | base | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.312<br>latency_p95 0.368<br>answer_chars 28.600 | — | [base](../eval/results/6e41cc21900dadc3dbaade7cabad6f5c.json) |

</details>
