"""Детерминированный кэш результатов прогонов.

Ключ складывается из того, что меняет результат: вариант (имя + канонизированные
оверрайды), хеш золотого набора, набор метрик, отпечаток окружения и параметры
прогона (runner, limit, with_answers, repeats, concurrency). Пока ничего из этого
не изменилось, повторный прогон возвращает сохранённый результат — base снимается
один раз и переиспользуется в A/B вместо повторной пересъёмки.

Хранилище — JSON в ``eval/results/<key>.json``. Каталог в gitignore: результаты
воспроизводимы из исходников, а объёмные артефакты в репозиторий не идут.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# При прямом запуске (`uv run python eval/cache.py`) в sys.path попадает
# каталог eval, а не корень репозитория, и импорт `eval` падает.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.config import Environment, RunConfig, Variant
from eval.result import RunResult, Verdict

logger = logging.getLogger("eval.cache")

#: Куда складываются результаты по умолчанию.
DEFAULT_DIR = Path(__file__).resolve().parent / "results"


def canonical_settings(variant: Variant) -> str:
    """Каноническое представление оверрайдов настроек.

    Args:
        variant: Вариант.

    Returns:
        str: JSON оверрайдов с сортировкой ключей.
    """
    return json.dumps(variant.settings, ensure_ascii=False, sort_keys=True)


def file_hash(path: str | Path) -> str:
    """SHA-256 содержимого файла.

    Args:
        path: Путь к файлу.

    Returns:
        str: Полный hex-хеш содержимого.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_version(metrics: Iterable[Any]) -> str:
    """Версия набора метрик — сортированный список их ключей.

    Args:
        metrics: Метрики (объекты с атрибутом ``key``).

    Returns:
        str: Ключи, соединённые ``+``; пустая строка, если метрик нет.
    """
    return "+".join(sorted(metric.key for metric in metrics))


def _config_digest(config: RunConfig) -> dict[str, Any]:
    """Канонический словарь параметров прогона для ключа.

    Args:
        config: Параметры прогона.

    Returns:
        dict[str, Any]: Параметры, влияющие на результат.
    """
    return {
        "runner": config.runner,
        "limit": config.limit,
        "with_answers": config.with_answers,
        "repeats": config.repeats,
        "concurrency": list(config.concurrency),
        "seed": config.seed,
        "bootstraps": config.bootstraps,
        "sample_index": config.sample_index,
        "graph": config.graph,
    }


def cache_key(
    variant: Variant,
    dataset: str,
    metrics: Iterable[Any],
    environment: Environment,
    config: RunConfig,
) -> str:
    """Считает детерминированный ключ результата.

    Args:
        variant: Вариант системы.
        dataset: Хеш золотого набора (см. :func:`file_hash`).
        metrics: Метрики замера.
        environment: Отпечаток окружения.
        config: Параметры прогона.

    Returns:
        str: SHA-256 (32 hex-символа) от всех входящих.
    """
    payload = {
        "variant": variant.identity(),
        "dataset": dataset,
        "metrics": metrics_version(metrics),
        "environment": environment.fingerprint(),
        "config": _config_digest(config),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_entry(
    key: str,
    variant: Variant,
    result: RunResult,
    environment: Environment,
    dataset: str,
    metrics: Iterable[Any],
    config: RunConfig,
) -> dict[str, Any]:
    """Собирает содержимое файла кэша.

    Args:
        key: Ключ результата.
        variant: Вариант.
        result: Результат прогона.
        environment: Отпечаток окружения.
        dataset: Хеш золотого набора.
        metrics: Метрики замера.
        config: Параметры прогона.

    Returns:
        dict[str, Any]: Полная запись кэша.
    """
    return {
        "key": key,
        "variant": {
            "name": variant.name,
            "description": variant.description,
            "settings": variant.settings,
        },
        "runner": config.runner,
        "environment": environment.fingerprint(),
        "environment_desc": environment.describe(),
        "dataset": dataset,
        "metrics": metrics_version(metrics),
        "config": _config_digest(config),
        "graph": config.graph,
        "created_at": result.created_at,
        "result": result.to_dict(),
    }


def ab_cache_key(
    base_variant: Variant,
    variant: Variant,
    dataset: str,
    metrics: Iterable[Any],
    environment: Environment,
    config: RunConfig,
) -> str:
    """Считает детерминированный ключ для A/B сравнения.

    Args:
        base_variant: Базовый вариант.
        variant: Сравниваемый вариант.
        dataset: Хеш золотого набора.
        metrics: Метрики замера.
        environment: Отпечаток окружения.
        config: Параметры прогона.

    Returns:
        str: SHA-256 (32 hex-символа) от пары вариантов и контекста.
    """
    payload = {
        "base_variant": base_variant.identity(),
        "variant": variant.identity(),
        "dataset": dataset,
        "metrics": metrics_version(metrics),
        "environment": environment.fingerprint(),
        "config": _config_digest(config),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "ab_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_ab_entry(
    key: str,
    base_variant: Variant,
    variant: Variant,
    verdicts: list[Verdict],
    environment: Environment,
    dataset: str,
    metrics: Iterable[Any],
    config: RunConfig,
) -> dict[str, Any]:
    """Собирает содержимое файла кэша A/B вердиктов.

    Args:
        key: Ключ результата.
        base_variant: Базовый вариант.
        variant: Сравниваемый вариант.
        verdicts: Список вердиктов.
        environment: Отпечаток окружения.
        dataset: Хеш золотого набора.
        metrics: Метрики замера.
        config: Параметры прогона.

    Returns:
        dict[str, Any]: Полная запись кэша A/B.
    """
    return {
        "key": key,
        "base_variant": base_variant.name,
        "variant": variant.name,
        "verdicts": [v.to_dict() for v in verdicts],
        "environment": environment.fingerprint(),
        "environment_desc": environment.describe(),
        "dataset": dataset,
        "metrics": metrics_version(metrics),
        "config": _config_digest(config),
        "graph": config.graph,
        "created_at": datetime.now(UTC).isoformat(),
    }


class ResultCache:
    """Хранилище результатов в JSON-файлах.

    Args:
        directory: Каталог кэша; по умолчанию ``eval/results``.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_DIR

    def path_for(self, key: str) -> Path:
        """Путь файла по ключу.

        Args:
            key: Ключ результата.

        Returns:
            Path: ``<каталог>/<key>.json``.
        """
        return self.directory / f"{key}.json"

    def load(self, key: str) -> RunResult | None:
        """Читает результат из кэша.

        Args:
            key: Ключ результата.

        Returns:
            RunResult: Сохранённый результат или ``None``, если его нет.
        """
        path = self.path_for(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunResult.from_dict(data["result"])

    def save(self, entry: dict[str, Any]) -> Path:
        """Пишет запись в кэш.

        Args:
            entry: Запись из :func:`build_entry`.

        Returns:
            Path: Куда записано.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(entry["key"])
        path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("результат сохранён: %s", path)
        return path

    def metadata(self) -> list[dict[str, Any]]:
        """Краткие описания всех сохранённых результатов.

        Returns:
            list[dict[str, Any]]: Ключ, вариант, раннер, окружение, дата.
        """
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "key": data["key"],
                    "variant": data["variant"]["name"],
                    "runner": data["runner"],
                    "environment": data["environment"],
                    "environment_desc": data.get("environment_desc", ""),
                    "dataset": data["dataset"],
                    "created_at": data["created_at"],
                }
            )
        return rows

    def clear(self) -> int:
        """Удаляет все сохранённые результаты.

        Returns:
            int: Сколько файлов удалено.
        """
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink()
            removed += 1
        logger.info("кэш очищен: удалено %d записей", removed)
        return removed

    # --- A/B вердикты ---

    def save_ab(self, entry: dict[str, Any]) -> Path:
        """Пишет запись A/B вердиктов в кэш.

        Args:
            entry: Запись из :func:`build_ab_entry`.

        Returns:
            Path: Куда записано.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(entry["key"])
        path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("A/B вердикты сохранены: %s", path)
        return path

    def load_ab(self, key: str) -> list[Verdict] | None:
        """Читает A/B вердикты из кэша.

        Args:
            key: Ключ A/B результата.

        Returns:
            list[Verdict] | None: Список вердиктов или None, если нет.
        """
        path = self.path_for(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Verdict.from_dict(v) for v in data.get("verdicts", [])]

    def list_ab(self) -> list[dict[str, Any]]:
        """Список сохранённых A/B вердиктов.

        Returns:
            list[dict]: Ключ, base, variant, окружение, дата, число вердиктов.
        """
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("ab_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "key": data["key"],
                    "base_variant": data.get("base_variant", ""),
                    "variant": data.get("variant", ""),
                    "environment": data["environment"],
                    "environment_desc": data.get("environment_desc", ""),
                    "dataset": data["dataset"],
                    "created_at": data["created_at"],
                    "verdicts_count": len(data.get("verdicts", [])),
                }
            )
        return rows

    def clear_ab(self) -> int:
        """Удаляет только A/B вердикты из кэша.

        Returns:
            int: Сколько файлов удалено.
        """
        removed = 0
        for path in self.directory.glob("ab_*.json"):
            path.unlink()
            removed += 1
        logger.info("A/B кэш очищен: удалено %d записей", removed)
        return removed


def _print_list(rows: list[dict[str, Any]]) -> None:
    """Печатает список результатов в виде таблицы.

    Args:
        rows: Метаданные из :meth:`ResultCache.metadata`.
    """
    if not rows:
        print("кэш пуст")
        return
    header = f"{'ключ':<34}{'вариант':<20}{'раннер':<14}{'окружение':<18}создан"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['key']:<34}{row['variant']:<20}{row['runner']:<14}"
            f"{row['environment']:<18}{row['created_at']}"
        )


def _print_ab_list(rows: list[dict[str, Any]]) -> None:
    """Печатает список A/B вердиктов в виде таблицы.

    Args:
        rows: Метаданные из :meth:`ResultCache.list_ab`.
    """
    if not rows:
        print("A/B кэш пуст")
        return
    header = (
        f"{'ключ':<36}{'base':<16}{'variant':<20}"
        f"{'окружение':<18}{'вердиктов':>8} создан"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['key']:<36}{row['base_variant']:<16}{row['variant']:<20}"
            f"{row['environment']:<18}{row['verdicts_count']:>8} {row['created_at']}"
        )


def main() -> int:
    """Точка входа управления кэшем.

    Returns:
        int: Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default=str(DEFAULT_DIR), help="каталог кэша (по умолчанию results)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="список сохранённых результатов")
    sub.add_parser("clear", help="удалить все сохранённые результаты")
    sub.add_parser("ab-list", help="список A/B вердиктов")
    sub.add_parser("ab-clear", help="удалить A/B вердикты")
    args = parser.parse_args()

    cache = ResultCache(args.dir)
    if args.command == "list":
        _print_list(cache.metadata())
    elif args.command == "clear":
        cache.clear()
    elif args.command == "ab-list":
        _print_ab_list(cache.list_ab())
    elif args.command == "ab-clear":
        cache.clear_ab()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
