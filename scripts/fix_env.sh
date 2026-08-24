#!/usr/bin/env bash
# Приводит deploy/.env к виду, пригодному и для стенда, и для запуска с хоста.
#
# Адреса внутри docker-стека подставляет сам compose, а на хосте верны
# умолчания из settings.py. Раньше это было безразлично: файл читал только
# compose. Теперь его читают и запуски с хоста (замер, индексация,
# verify.sh), и строка вида QDRANT_URL=http://qdrant:6333 ломает их - имя
# `qdrant` с хоста не разрешается.
#
# Скрипт закомментирует такие строки, сделав рядом копию .env.bak.
#
#     bash scripts/fix_env.sh            # показать, что будет изменено
#     bash scripts/fix_env.sh --apply    # применить

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FILE="deploy/.env"
KEYS='QDRANT_URL|OLLAMA_LOCAL_BASE_URL|POSTGRES_HOST|AEGRA_URL|GUI_DATABASE_URL|RERANK_ENDPOINT'

[ -f "$FILE" ] || { echo "нет файла $FILE: cp deploy/.env.example $FILE"; exit 1; }

found=$(grep -nE "^\s*($KEYS)=" "$FILE" || true)
if [ -z "$found" ]; then
  echo "В $FILE нет адресов, мешающих запуску с хоста."
else
  echo "Эти строки верны только внутри стека и мешают запуску с хоста:"
  echo "$found" | sed 's/^/  /'
fi

# QDRANT_COLLECTION задаёт имя АЛИАСА. Если туда записать имя готовой
# коллекции (aist_chunks_v2), поколение соберётся как aist_chunks_v2_v1, а
# переключение алиаса упадёт: Qdrant не даст создать алиас с именем
# существующей коллекции.
coll=$(grep -E "^\s*QDRANT_COLLECTION=" "$FILE" | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" | xargs)
if printf '%s' "$coll" | grep -qE '_v[0-9]+$'; then
  echo
  echo "QDRANT_COLLECTION=$coll — это имя поколения, а нужно имя алиаса."
  echo "  поколения называются <алиас>_v1, <алиас>_v2 и выбираются флагом --generation"
  echo "  правильно: QDRANT_COLLECTION=$(printf '%s' "$coll" | sed -E 's/_v[0-9]+$//')"
  alias_wrong=1
else
  alias_wrong=0
fi

if grep -qE "^\s*RERANK_ENABLED=1" "$FILE"; then
  rerank_ok=1
else
  rerank_ok=0
  echo
  echo "Переоценка выключена, а она даёт основной прирост качества:"
  echo "  hit@1 0,561 без неё против 0,732 с ней"
fi

if [ "${1:-}" != "--apply" ]; then
  echo
  echo "Показан предпросмотр. Применить: bash scripts/fix_env.sh --apply"
  exit 0
fi

cp "$FILE" "$FILE.bak"
if [ "$alias_wrong" = "1" ]; then
  fixed=$(printf '%s' "$coll" | sed -E 's/_v[0-9]+$//')
  perl -i -pe "s/^(\s*)QDRANT_COLLECTION=.*/\$1QDRANT_COLLECTION=\"$fixed\"/" "$FILE"
  echo "  QDRANT_COLLECTION приведён к имени алиаса: $fixed"
fi
# shellcheck disable=SC2016
perl -i -pe "s/^(\s*)($KEYS)=/\$1# адрес задаёт compose, на хосте мешает: \$2=/" "$FILE"
[ "$rerank_ok" = "0" ] && printf '\n# Переоценка: основной источник качества (hit@1 0,732 против 0,561).\nRERANK_ENABLED=1\n' >> "$FILE"
echo
echo "Готово. Копия прежнего файла: $FILE.bak"
echo "Проверьте итог: bash scripts/verify.sh --full"
