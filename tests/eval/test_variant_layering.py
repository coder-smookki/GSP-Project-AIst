"""Тесты наложения оверрайдов варианта на настройки стенда.

Замер сравнивает варианты, поэтому настройки собираются из двух слоёв:
файл стенда задаёт базу, оверрайды варианта перекрывают названные ключи.
Пока файл `deploy/.env` не читался при запуске с хоста, базой молча
становились умолчания, и замер без `--settings` относился не к той
конфигурации, которую выставил человек.
"""

from __future__ import annotations

import pytest

from eval.config import Variant
from eval.variants import make_settings
from infrastructure.config.settings import Settings


@pytest.fixture
def _stack_env(tmp_path, monkeypatch):
    """Готовит временный корень с файлом настроек стенда.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Штатная фикстура подмены значений.

    Returns:
        Path: Каталог, ставший текущим.
    """
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy" / ".env").write_text(
        "RETRIEVAL_PREFETCH=10\nRETRIEVAL_RRF_K=40\nRETRIEVAL_MIN_RELEVANCE=0.45\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ("deploy/.env", ".env"))
    return tmp_path


def test_empty_variant_takes_stack_settings(_stack_env) -> None:
    """Замер без оверрайдов идёт с настройками стенда, а не с умолчаниями."""
    settings = make_settings(Variant(name="пустой", settings={}))
    assert settings.retrieval_prefetch == 10
    assert settings.retrieval_rrf_k == 40
    assert settings.retrieval_min_relevance == 0.45


def test_override_wins_only_for_named_keys(_stack_env) -> None:
    """Оверрайд перекрывает названное, остальное берётся из файла."""
    settings = make_settings(
        Variant(name="свой prefetch", settings={"retrieval_prefetch": 30})
    )
    assert settings.retrieval_prefetch == 30
    assert settings.retrieval_rrf_k == 40


def test_defaults_apply_without_stack_file(tmp_path, monkeypatch) -> None:
    """Без файла действуют умолчания: это допустимо, но должно быть видно."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ("deploy/.env", ".env"))
    settings = make_settings(Variant(name="без файла", settings={}))
    assert settings.retrieval_prefetch == 30
    assert settings.retrieval_rrf_k == 60
