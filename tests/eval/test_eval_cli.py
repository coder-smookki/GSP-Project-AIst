"""Тесты единого CLI бенчмаркинга.

Проверяется только то, что не трогает живую инфраструктуру: дерево подкоманд,
разбор аргументов и управление кэшем. Сами прогоны покрыты тестами модулей.
"""

from __future__ import annotations

import argparse

from eval.cache import ResultCache, build_entry
from eval.cli import _cmd_cache, build_parser
from eval.config import RunConfig, Variant
from tests.eval.test_eval_cache import ENV, METRICS, key_for
from tests.eval.test_eval_stats import make_result


class TestBuildParser:
    """Дерево подкоманд."""

    def test_subcommands(self) -> None:
        """Все харнессы доступны как подкоманды."""
        parser = build_parser()
        subparser_actions = [
            action
            for action in parser._subparsers._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        subcommands = next(iter(subparser_actions)).choices
        assert set(subcommands) == {"run", "ab", "bench", "tune", "cache", "report"}

    def test_run_parses_variant_settings(self) -> None:
        """run принимает имя и оверрайды варианта."""
        args = build_parser().parse_args(
            ["run", "--variant", "top6", "--settings", '{"retrieval_top_k": 6}']
        )
        assert args.command == "run"
        assert args.variant == "top6"
        assert args.settings == '{"retrieval_top_k": 6}'

    def test_ab_requires_variants(self) -> None:
        """ab без файла вариантов — ошибка."""
        parser = build_parser()
        try:
            parser.parse_args(["ab"])
        except SystemExit:
            return
        raise AssertionError("ожидалась ошибка парсинга")


class TestCmdCache:
    """Управление кэшем через CLI."""

    def test_list_shows_entries(self, tmp_path, capsys) -> None:
        """list печатает сохранённые результаты."""
        cache = ResultCache(tmp_path)
        result = make_result([1, 2])
        entry = build_entry(
            key_for(), Variant(name="base"), result, ENV, "abc", METRICS, RunConfig()
        )
        cache.save(entry)
        args = argparse.Namespace(cache_command="list", dir=str(tmp_path))
        assert _cmd_cache(args) == 0
        out = capsys.readouterr().out
        assert "base" in out

    def test_clear_removes_entries(self, tmp_path) -> None:
        """clear удаляет все записи."""
        cache = ResultCache(tmp_path)
        result = make_result([1, 2])
        entry = build_entry(
            key_for(), Variant(name="base"), result, ENV, "abc", METRICS, RunConfig()
        )
        cache.save(entry)
        args = argparse.Namespace(cache_command="clear", dir=str(tmp_path))
        assert _cmd_cache(args) == 0
        assert ResultCache(tmp_path).metadata() == []
