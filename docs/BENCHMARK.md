# Бенчмарк качества

Измерение качества поиска и ответов на золотом наборе, A/B-сравнение
вариантов и история результатов. Этот документ генерируется командой
`uv run eval report` из кэша `eval/results/`.

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

## A/B-тест

Варианты сравниваются на одном золотом наборе, поэтому выборки парные.
Категориальные метрики (`hit@k`, `refusal_rate`, `not_found`) — точный тест
МакНемара по расходящимся парам; непрерывные (`mrr`, латенси) — знаково-
ранговый критерий Уилкоксона. Без scipy обе заменяются парным пермутационным
тестом. Вердикт значим при `p < 0.05`.

## История результатов

Каждая строка — один прогон: группа вариантов, снятых на одном наборе,
окружении и конфиге.

| дата | окружение | варианты | метрики | вердикт | кэш |
| --- | --- | --- | --- | --- | --- |
| 2026-08-14 11:35 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=выкл | rerank_off | hit@1 0.561<br>hit@3 0.683<br>hit@5 0.780<br>mrr 0.645<br>not_found 0.220<br>refusal_rate 0.500<br>latency_median 0.127<br>latency_p95 0.140<br>answer_chars 0.000 | — | [rerank_off](../eval/results/2258167afa8a03091d768f014c5c22c5.json) |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base, rerank_on | hit@1 base=0.732 / rerank_on=0.732<br>hit@3 base=0.805 / rerank_on=0.805<br>hit@5 base=0.854 / rerank_on=0.854<br>mrr base=0.776 / rerank_on=0.776<br>not_found base=0.146 / rerank_on=0.146<br>refusal_rate base=0.500 / rerank_on=0.500<br>latency_median base=0.294 / rerank_on=0.279<br>latency_p95 base=0.327 / rerank_on=0.307<br>answer_chars base=0.000 / rerank_on=0.000 | rerank_on (лучше: latency_median, latency_p95) | [base](../eval/results/5f17c854b73bdb8e8616313e12e008a6.json)<br>[rerank_on](../eval/results/659e3d103f3e9147a4715ce9118010ab.json) |
| 2026-08-13 22:16 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | gptoss_self_r0g0 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.690<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.154<br>latency_p95 0.158<br>answer_chars 28.500 | — | [gptoss_self_r0g0](../eval/results/36539bc1d8aaad1b822e33cb4f3579cb.json) |
| 2026-08-13 22:14 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | gptoss_self_r0g1 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.690<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.154<br>latency_p95 0.159<br>answer_chars 28.400 | — | [gptoss_self_r0g1](../eval/results/140f53776bc9ae193865aa3db22b7672.json) |
| 2026-08-13 22:12 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | gptoss_self_r1g1 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.316<br>latency_p95 0.391<br>answer_chars 28.500 | — | [gptoss_self_r1g1](../eval/results/4352a10b17b4dce44e5d85ea734d7a78.json) |
| 2026-08-13 22:10 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | gptoss_gemma_r0g0 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.690<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.154<br>latency_p95 0.160<br>answer_chars 28.500 | — | [gptoss_gemma_r0g0](../eval/results/605eefa2a64d13e07897f24980cad3f0.json) |
| 2026-08-13 22:09 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | gptoss_gemma_r0g1 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.691<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.154<br>latency_p95 0.159<br>answer_chars 28.400 | — | [gptoss_gemma_r0g1](../eval/results/2f8c8f59dae7ee3a726181a0b2437d9e.json) |
| 2026-08-13 22:08 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | gptoss_gemma_r1g0 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.314<br>latency_p95 0.399<br>answer_chars 28.500 | — | [gptoss_gemma_r1g0](../eval/results/6b420e5e191df6a4566617e3cf793350.json) |
| 2026-08-13 22:07 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | gptoss_gemma_r1g1 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.315<br>latency_p95 1.613<br>answer_chars 28.500 | — | [gptoss_gemma_r1g1](../eval/results/54547c141d0c2795a5cea75d22011360.json) |
| 2026-08-13 22:05 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | qwen36_self_r0g0 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.690<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.155<br>latency_p95 0.161<br>answer_chars 28.500 | — | [qwen36_self_r0g0](../eval/results/543ac6b336b7255c1134a1a3bd31ff01.json) |
| 2026-08-13 22:05 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | qwen36_self_r0g1 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.690<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.155<br>latency_p95 0.162<br>answer_chars 28.500 | — | [qwen36_self_r0g1](../eval/results/c868f329c0b564d72d242c76697961d9.json) |
| 2026-08-13 22:04 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | qwen36_self_r1g0 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.326<br>latency_p95 0.604<br>answer_chars 28.500 | — | [qwen36_self_r1g0](../eval/results/b5d8d5fd1e6f6c656d2d51e399efdea7.json) |
| 2026-08-13 22:04 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | qwen36_self_r1g1 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.337<br>latency_p95 2.150<br>answer_chars 28.500 | — | [qwen36_self_r1g1](../eval/results/c691de21de57c2db531f556b82807acf.json) |
| 2026-08-13 22:02 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | qwen36_gemma_r0g0 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.691<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.166<br>latency_p95 0.175<br>answer_chars 28.400 | — | [qwen36_gemma_r0g0](../eval/results/f5c195c2ceadf9865f8a9a7965883c0b.json) |
| 2026-08-13 22:02 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=выкл | qwen36_gemma_r0g1 | hit@1 0.634<br>hit@3 0.683<br>hit@5 0.829<br>mrr 0.690<br>not_found 0.171<br>refusal_rate 1.000<br>latency_median 0.169<br>latency_p95 0.185<br>answer_chars 28.400 | — | [qwen36_gemma_r0g1](../eval/results/7dda9ab1528339887b14f7e0d6c24318.json) |
| 2026-08-13 22:01 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | qwen36_gemma_r1g0 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.341<br>latency_p95 3.530<br>answer_chars 28.400 | — | [qwen36_gemma_r1g0](../eval/results/3a8a80de6f58fa2c691c9f638f5e3ba3.json) |
| 2026-08-13 22:00 | llm=qwen3.6:35b-a3b-nvfp4, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | qwen36_gemma_r1g1 | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.324<br>latency_p95 1.235<br>answer_chars 28.500 | — | [qwen36_gemma_r1g1](../eval/results/3610ef0e591fee5fcd6f94fc8cabaaf9.json) |
| 2026-08-13 21:59 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | base | hit@1 0.756<br>hit@3 0.829<br>hit@5 0.854<br>mrr 0.799<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.321<br>latency_p95 0.361<br>answer_chars 28.700 | — | [base](../eval/results/90b2348754de0b0377a6cb07984a52a3.json) |
| 2026-08-13 21:56 | llm=gpt-oss:20b, emb=qwen3-embedding:4b, top_k=5, rerank=вкл | base | hit@1 0.667<br>hit@3 1.000<br>hit@5 1.000<br>mrr 0.833<br>not_found 0.000<br>refusal_rate 1.000<br>latency_median 0.324<br>latency_p95 6.288<br>answer_chars 28.667 | — | [base](../eval/results/c2d75d629cf09526afa5849f5150eba6.json) |
| 2026-08-13 21:44 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=выкл | gptoss_self_r0g0 | hit@1 0.561<br>hit@3 0.707<br>hit@5 0.780<br>mrr 0.647<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.133<br>latency_p95 0.141<br>answer_chars 28.700 | — | [gptoss_self_r0g0](../eval/results/3f3b2c8ce6f4a8bbf3656c72bfbf1a56.json) |
| 2026-08-13 21:42 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=выкл | gptoss_self_r0g1 | hit@1 0.561<br>hit@3 0.683<br>hit@5 0.780<br>mrr 0.646<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.134<br>latency_p95 0.142<br>answer_chars 28.700 | — | [gptoss_self_r0g1](../eval/results/6d97e67b29a8a1beb5e2a22df3ab56e2.json) |
| 2026-08-13 21:40 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | gptoss_self_r1g1 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.291<br>latency_p95 0.340<br>answer_chars 28.700 | — | [gptoss_self_r1g1](../eval/results/668ec84eeb36c749928a7aa9e6509b2f.json) |
| 2026-08-13 21:38 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=выкл | gptoss_gemma_r0g0 | hit@1 0.561<br>hit@3 0.707<br>hit@5 0.780<br>mrr 0.647<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.128<br>latency_p95 0.133<br>answer_chars 28.700 | — | [gptoss_gemma_r0g0](../eval/results/41e8a5571b2959a9b25ea8490156efc2.json) |
| 2026-08-13 21:37 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=выкл | gptoss_gemma_r0g1 | hit@1 0.561<br>hit@3 0.732<br>hit@5 0.780<br>mrr 0.649<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.128<br>latency_p95 0.137<br>answer_chars 28.700 | — | [gptoss_gemma_r0g1](../eval/results/34b9db8b137b48aee5a9eccda1d4fadd.json) |
| 2026-08-13 21:36 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | gptoss_gemma_r1g0 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.284<br>latency_p95 0.334<br>answer_chars 28.700 | — | [gptoss_gemma_r1g0](../eval/results/2a94b894eb26b7a0b11bbee1f371fcc5.json) |
| 2026-08-13 21:35 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | gptoss_gemma_r1g1 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.285<br>latency_p95 0.376<br>answer_chars 28.700 | — | [gptoss_gemma_r1g1](../eval/results/8cee1be28e8d6bd607a7b89639c2b558.json) |
| 2026-08-13 21:32 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=выкл | qwen36_self_r0g0 | hit@1 0.561<br>hit@3 0.683<br>hit@5 0.780<br>mrr 0.645<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.129<br>latency_p95 0.142<br>answer_chars 28.800 | — | [qwen36_self_r0g0](../eval/results/cf800f9275c9b067fce58b81d2be5fa4.json) |
| 2026-08-13 21:32 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=выкл | qwen36_self_r0g1 | hit@1 0.561<br>hit@3 0.707<br>hit@5 0.780<br>mrr 0.648<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.128<br>latency_p95 0.134<br>answer_chars 28.700 | — | [qwen36_self_r0g1](../eval/results/8454c4b5de2a28b5dff616064a755cd5.json) |
| 2026-08-13 21:31 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=вкл | qwen36_self_r1g0 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.286<br>latency_p95 0.334<br>answer_chars 28.800 | — | [qwen36_self_r1g0](../eval/results/4892e4afbba87991cc20cecbaca0e1ac.json) |
| 2026-08-13 21:30 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=вкл | qwen36_self_r1g1 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.290<br>latency_p95 0.463<br>answer_chars 28.700 | — | [qwen36_self_r1g1](../eval/results/efbbf2c184b083dd4060c89bd417fd75.json) |
| 2026-08-13 21:29 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=выкл | qwen36_gemma_r0g0 | hit@1 0.561<br>hit@3 0.732<br>hit@5 0.780<br>mrr 0.649<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.128<br>latency_p95 0.133<br>answer_chars 28.600 | — | [qwen36_gemma_r0g0](../eval/results/3d2249135d426beeece9d03823d2e577.json) |
| 2026-08-13 21:29 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=выкл | qwen36_gemma_r0g1 | hit@1 0.561<br>hit@3 0.683<br>hit@5 0.780<br>mrr 0.645<br>not_found 0.220<br>refusal_rate 1.000<br>latency_median 0.128<br>latency_p95 0.135<br>answer_chars 28.700 | — | [qwen36_gemma_r0g1](../eval/results/a59fa5723920437768183d63c299e5b0.json) |
| 2026-08-13 21:28 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=вкл | qwen36_gemma_r1g0 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.290<br>latency_p95 0.342<br>answer_chars 28.700 | — | [qwen36_gemma_r1g0](../eval/results/6866d69c828e8b84a30c6a33e74e1d40.json) |
| 2026-08-13 21:28 | llm=qwen3.6:35b-a3b-nvfp4, emb=bge-m3:567m, top_k=5, rerank=вкл | qwen36_gemma_r1g1 | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 1.000<br>latency_median 0.300<br>latency_p95 0.348<br>answer_chars 28.700 | — | [qwen36_gemma_r1g1](../eval/results/e56986b7daa088a07e4b233deb2da050.json) |
| 2026-08-13 20:53 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | hit@1 0.732<br>hit@3 0.805<br>hit@5 0.854<br>mrr 0.776<br>not_found 0.146<br>refusal_rate 0.900<br>latency_median 0.312<br>latency_p95 0.367<br>answer_chars 42.900 | — | [base](../eval/results/694011085ef1fedfdb18fdaabf9a5317.json) |


## A/B сравнения (закэшированные вердикты)

Таблица показывает все сохранённые A/B вердикты. «Δ» — разница variant - base.
Вердикт: «лучше» — значимое улучшение (p < 0.05), «хуже» — ухудшение, «без разницы» — нет значимых отличий.

| дата | окружение | base | variant | метрика | base | variant | Δ | p-value | вердикт | тест |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | hit@1 | 0.732 | 0.732 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | hit@3 | 0.805 | 0.805 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | hit@5 | 0.854 | 0.854 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | mrr | 0.776 | 0.776 | +0.000 | 1.000 | без разницы | wilcoxon |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | not_found | 0.146 | 0.146 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | refusal_rate | 0.500 | 0.500 | +0.000 | 1.000 | без разницы | mcnemar |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | latency_median | 0.294 | 0.279 | -0.015 | 0.000 | лучше | wilcoxon |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | latency_p95 | 0.327 | 0.307 | -0.020 | 0.000 | лучше | wilcoxon |
| 2026-08-14 11:34 | llm=gpt-oss:20b, emb=bge-m3:567m, top_k=5, rerank=вкл | base | rerank_on | answer_chars | 0.000 | 0.000 | +0.000 | 1.000 | без разницы | wilcoxon |
