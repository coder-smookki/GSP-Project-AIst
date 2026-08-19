#!/usr/bin/env bash
# Проверка изменений на целевой машине.
#
# Сценарий для коллеги: прогоняет всё, что можно проверить машинно, и
# печатает результат так, чтобы расхождение было видно сразу. Ничего не
# меняет на диске, кроме файлов отчётов в eval/results/.
#
# Запуск из корня репозитория:
#     bash scripts/verify.sh            # без живых сервисов: тесты и линтер
#     bash scripts/verify.sh --full     # плюс индексация, качество и нагрузка
#
# Для --full нужны поднятый стек (docker compose up -d) и скачанные модели.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

pass=0
fail=0

step() { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }
ok()   { printf "  \033[32m[ок]\033[0m %s\n" "$1"; pass=$((pass+1)); }
bad()  { printf "  \033[31m[!!]\033[0m %s\n" "$1"; fail=$((fail+1)); }

step "1. Статический контроль"
if $PY -m ruff check . >/tmp/aist_ruff.txt 2>&1; then
  ok "линтер чист"
else
  bad "линтер ругается:"; tail -6 /tmp/aist_ruff.txt | sed 's/^/       /'
fi

step "2. Тесты без живых сервисов"
if $PY -m pytest -q -m "not llm and not embeddings and not qdrant and not integration" \
     >/tmp/aist_tests.txt 2>&1; then
  ok "$(grep -E "passed|failed" /tmp/aist_tests.txt | tail -1)"
else
  # Именно сводка и список упавших: в хвосте файла обычно шум от логгера
  # Prefect при остановке временного сервера, и по нему причину не понять.
  bad "тесты упали:"
  grep -E "^FAILED|^ERROR" /tmp/aist_tests.txt | head -6 | sed 's/^/       /'
  grep -E "passed|failed" /tmp/aist_tests.txt | tail -1 | sed 's/^/       /'
  echo "       полный вывод: /tmp/aist_tests.txt"
fi

step "3. Проверки исправленных дефектов"

# Панель источников: блок custom обязан стоять ВЫШЕ отсева не-updates,
# иначе события до него не доходят и панель не рисуется никогда.
custom_line=$(grep -n 'if chunk.event == "custom"' gui/chainlit_app.py | head -1 | cut -d: -f1)
guard_line=$(grep -n 'if chunk.event != "updates"' gui/chainlit_app.py | head -1 | cut -d: -f1)
if [ -n "$custom_line" ] && [ -n "$guard_line" ] && [ "$custom_line" -lt "$guard_line" ]; then
  ok "панель источников: обработчик custom выше отсева (строки $custom_line < $guard_line)"
else
  bad "панель источников: обработчик custom недостижим"
fi

# Стриминг в режиме чата: включение стрима должно ловить и узел plan.
if grep -q 'node_name in ("plan", "finalize")' gui/chainlit_app.py; then
  ok "стриминг включается после планировщика (работает и в режиме чата)"
else
  bad "стриминг включается только после finalize: в режиме чата ответ не стримится"
fi

# Состояние индекса: неоднозначный отпечаток обязан вести к переиндексации.
if grep -q "документы с несколькими отпечатками" infrastructure/ingest/incremental.py; then
  ok "состояние индекса: оборванная загрузка приводит к переиндексации"
else
  bad "состояние индекса: документ с оборванной загрузкой может быть пропущен навсегда"
fi

# Слияние ветвей: доли вместо общей сортировки по несопоставимым скорам.
if grep -q "_merge_branches" agent/graph.py; then
  ok "слияние ветвей: каждому подзапросу выделяется своя доля"
else
  bad "слияние ветвей: общая сортировка по скору, ответ на часть вопроса может выпасть"
fi

# Лицензии: в NOTICE не должно остаться незаполненных полей и опечаток.
if ! grep -qE "^\- \*\*(Название|Автор / правообладатель):\*\* \?$" NOTICE.md \
   && ! grep -q "aigra-api" NOTICE.md; then
  ok "NOTICE заполнен, несуществующих пакетов нет"
else
  bad "NOTICE: незаполненные поля или несуществующий пакет"
fi

# Сбои OpenAI-совместимых серверов (vLLM, llama.cpp) обязаны перехватываться:
# SDK заворачивает их в свои классы и httpx наружу не отдаёт.
if grep -q "from openai import APIError" agent/graph.py; then
  ok "сбои vLLM и llama.cpp дают понятное сообщение, а не трассировку"
else
  bad "сбой OpenAI-совместимого сервера пролетит мимо перехвата"
fi

# Временный сбой сервиса переоценки не должен выключать её до перезапуска.
if grep -q "REMOTE_COOLDOWN_SECONDS" infrastructure/retrieval/rerank.py; then
  ok "переоценка возвращается сама после паузы, качество не падает молча"
else
  bad "одна ошибка выключает переоценку навсегда: hit@1 упадёт с 0,829 до 0,561"
fi

step "4. Клиенты провайдеров (issue #33)"
for prov in vllm ollama-local llamacpp; do
  if $PY -c "
from infrastructure.config.settings import Settings
from infrastructure.config.model_registry import build_model_client
build_model_client(Settings(llm_provider='$prov'), 'модель')
" >/dev/null 2>&1; then
    ok "провайдер $prov собирается"
  else
    bad "провайдер $prov не собирается"
  fi
done
if $PY -c "
from infrastructure.config.settings import Settings
try:
    Settings(llm_provider='несуществующий')
except Exception as e:
    assert 'несуществующий' in str(e) or 'llm_provider' in str(e)
else:
    raise SystemExit(1)
" >/dev/null 2>&1; then
  ok "неизвестный провайдер отвергается понятной ошибкой"
else
  bad "неизвестный провайдер даёт невнятный сбой"
fi

step "5. Этапы выполнения (issue #34)"
if [ -f gui/stages.py ] && grep -q "parse_stage" gui/contract.py; then
  ok "этапы выполнения подключены"
else
  bad "этапы выполнения не найдены"
fi

if [ "$FULL" = "0" ]; then
  printf "\n\033[1mИтог:\033[0m успешно %d, замечаний %d\n" "$pass" "$fail"
  printf "Полная проверка с живыми сервисами: bash scripts/verify.sh --full\n"
  [ "$fail" = "0" ] || exit 1
  exit 0
fi

step "6. Живые сервисы"
# Проверяем до долгих шагов: иначе индексация или замер упадут через минуты
# ожидания, и причина утонет в трассировке.
qdrant_url="${QDRANT_URL:-http://localhost:6333}"
ollama_url="${OLLAMA_LOCAL_BASE_URL:-http://localhost:11434}"
services_ok=1
if curl -sf --max-time 4 "$qdrant_url/collections" >/dev/null 2>&1; then
  ok "Qdrant отвечает ($qdrant_url)"
else
  bad "Qdrant недоступен ($qdrant_url): cd deploy && docker compose up -d"
  services_ok=0
fi
if curl -sf --max-time 4 "$ollama_url/api/tags" >/dev/null 2>&1; then
  ok "Ollama отвечает ($ollama_url)"
else
  bad "Ollama недоступна ($ollama_url): запустите ollama serve"
  services_ok=0
fi
if [ "$services_ok" = "0" ]; then
  printf "\n\033[1mИтог:\033[0m успешно %d, замечаний %d\n" "$pass" "$fail"
  printf "Живые сервисы не подняты, шаги 7-9 пропущены.\n"
  exit 1
fi

# Размерность вектора прошита в схему коллекции. Если .env указывает на
# эмбеддер другой размерности, индексация пропустит документы как
# неизменившиеся, а упадёт всё позже - на запросе, с сообщением про
# «Vector dimension error», по которому причину не угадать.
# Печатаем конфигурацию, с которой реально пойдёт замер. Раньше её
# приходилось угадывать: файл deploy/.env запуск с хоста не видел, молча
# применялись умолчания, и цифры относились не к той настройке, которую
# человек выставил.
$PY - <<'PYCHECK'
import json
import sys
import urllib.request

from infrastructure.config.settings import get_settings

s = get_settings()
print("  конфигурация замера:")
print(f"    модель ответа      {s.llm_model}")
print(f"    эмбеддер           {s.embedding_model}")
print(f"    коллекция          {s.qdrant_collection}")
print(f"    фрагментов         {s.retrieval_top_k}")
print(f"    prefetch / rrf     {s.retrieval_prefetch} / {s.retrieval_rrf_k}")
print(f"    порог близости     {s.retrieval_min_relevance}")
print(f"    переоценка         {'вкл' if s.rerank_enabled else 'выкл'}")

# Семейство qwen3-embedding требует инструкцию к запросу: без неё карточка
# модели обещает падение качества на 1-5 %. Забыть легко, а заметить нечем -
# поиск работает, просто хуже.
if "qwen3-embedding" in s.embedding_model and not s.embedding_query_instruction:
    print("    ВНИМАНИЕ: инструкция к запросу пуста, а qwen3-embedding её требует")
    print("      образец значения - в deploy/.env.example")
elif s.embedding_query_instruction:
    print(f"    инструкция запроса {len(s.embedding_query_instruction)} симв.")

url = s.qdrant_url.rstrip("/")
try:
    with urllib.request.urlopen(
        f"{url}/collections/{s.qdrant_collection}", timeout=5
    ) as response:
        vectors = json.load(response)["result"]["config"]["params"]["vectors"]
    size = vectors.get("dense", vectors).get("size") if isinstance(vectors, dict) else None
    print(f"    размерность в индексе {size}")
except Exception:
    print(f"    коллекция {s.qdrant_collection} ещё не создана, соберём этим прогоном")
sys.exit(0)
PYCHECK

step "7. Индекс"
# Сценарий проверки не должен перестраивать индекс. Раньше здесь безусловно
# звался ingest --switch-alias без --generation, то есть всегда в поколение
# <алиас>_v1. Если это поколение собрано другим эмбеддером, прогон упирался
# в расхождение размерности, а если имя алиаса совпадало с именем готовой
# коллекции - в конфликт имён. Проверка обязана смотреть, а не менять.
index_state=$($PY - <<'PYSTATE'
import json
import urllib.request

from infrastructure.config.settings import get_settings

s = get_settings()
url = s.qdrant_url.rstrip("/")
try:
    with urllib.request.urlopen(
        f"{url}/collections/{s.qdrant_collection}", timeout=5
    ) as response:
        result = json.load(response)["result"]
    vectors = result["config"]["params"]["vectors"]
    size = vectors.get("dense", vectors).get("size") if isinstance(vectors, dict) else 0
    print(f"ready {result.get('points_count', 0)} {size}")
except Exception:
    print("missing 0 0")
PYSTATE
)
set -- $index_state
state="$1"; points="$2"; dim="$3"

if [ "$state" = "ready" ] && [ "$points" -gt 0 ]; then
  ok "индекс на месте: $points фрагментов, размерность $dim"
else
  bad "индекс пуст или недоступен — соберите его командой:"
  echo "       uv run python scripts/ingest.py --source files --generation N --switch-alias"
  echo "       (N — номер свободного поколения; существующие видны в Qdrant)"
fi

# Разбор корпуса проверяем всегда, но без записи: он от индекса не зависит.
if $PY scripts/ingest.py --source files --dry-run >/tmp/aist_ingest.txt 2>&1; then
  if grep -q "ошибок 0" /tmp/aist_ingest.txt; then
    ok "разбор корпуса без ошибок: $(grep -oE 'документов [0-9]+, чанков [0-9]+' /tmp/aist_ingest.txt | tail -1)"
  else
    bad "при разборе корпуса есть ошибки"
  fi
else
  bad "разбор корпуса не отработал:"; tail -5 /tmp/aist_ingest.txt | sed 's/^/       /'
fi

step "8. Качество на золотом наборе"
echo "  (может занять несколько минут)"
if $PY -m eval.cli run >/tmp/aist_eval.txt 2>&1; then
  grep -iE "hit@|mrr|отказ|not_found" /tmp/aist_eval.txt | head -8 | sed 's/^/       /'
  ok "замер качества выполнен"
else
  bad "замер качества не отработал:"; tail -8 /tmp/aist_eval.txt | sed 's/^/       /'
fi

step "9. Операционный замер"
# Замер переехал в пакет eval: scripts/benchmark.py больше не существует.
if $PY -m eval.cli bench --repeats 3 --concurrency 1 2 4 \
     --output eval/results/bench_verify.json >/tmp/aist_bench.txt 2>&1; then
  grep -E "медиана|ответов в час|запросов:" /tmp/aist_bench.txt | sed 's/^/       /'
  ok "операционный замер выполнен, подробности в eval/results/bench_verify.json"
else
  bad "операционный замер не отработал:"; tail -8 /tmp/aist_bench.txt | sed 's/^/       /'
fi

printf "\n\033[1mИтог:\033[0m успешно %d, замечаний %d\n" "$pass" "$fail"
[ "$fail" = "0" ] || exit 1
