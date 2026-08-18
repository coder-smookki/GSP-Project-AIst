"""Тесты чтения файла настроек с хоста.

Настройки стенда лежат в ``deploy/.env``, потому что оттуда их берёт docker
compose. Но путь в pydantic считается от текущего каталога, и запуск с хоста
из корня репозитория этот файл не видел: применялись умолчания, а человек
был уверен, что померил свою конфигурацию. Замер качества из-за этого шёл с
``prefetch=30, rrf=60, min_rel=0.55`` вместо выставленных ``10, 40, 0.45``.
"""

from __future__ import annotations

import pytest

from infrastructure.config.settings import Settings


@pytest.fixture
def _repo(tmp_path, monkeypatch):
    """Готовит временный каталог, похожий на корень репозитория.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Штатная фикстура подмены значений.

    Returns:
        Path: Корень временного репозитория.
    """
    (tmp_path / "deploy").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ("deploy/.env", ".env"))
    return tmp_path


def test_deploy_env_is_read_from_repo_root(_repo) -> None:
    """Настройки стенда применяются при запуске с хоста."""
    (_repo / "deploy" / ".env").write_text(
        "RETRIEVAL_PREFETCH=10\nRETRIEVAL_RRF_K=40\nRETRIEVAL_MIN_RELEVANCE=0.45\n",
        encoding="utf-8",
    )
    settings = Settings()
    assert settings.retrieval_prefetch == 10
    assert settings.retrieval_rrf_k == 40
    assert settings.retrieval_min_relevance == 0.45


def test_root_env_overrides_deploy(_repo) -> None:
    """Корневой файл перекрывает стендовый: так удобно пробовать варианты."""
    (_repo / "deploy" / ".env").write_text("RETRIEVAL_PREFETCH=10\n", encoding="utf-8")
    (_repo / ".env").write_text("RETRIEVAL_PREFETCH=30\n", encoding="utf-8")
    assert Settings().retrieval_prefetch == 30


def test_defaults_apply_without_any_file(_repo) -> None:
    """Без файлов действуют умолчания, и это не ошибка."""
    settings = Settings()
    assert settings.retrieval_prefetch == 30
    assert settings.qdrant_url == "http://localhost:6333"


def test_host_address_is_not_taken_from_stack_file(_repo) -> None:
    """Адрес Qdrant остаётся хостовым, даже если в файле стендовый.

    Внутри стека адрес подставляет compose, а на хосте он обязан остаться
    локальным: иначе запуск с хоста упрётся в неразрешимое имя `qdrant`.
    Поэтому сетевые адреса в шаблоне закомментированы.
    """
    (_repo / "deploy" / ".env").write_text(
        "QDRANT_COLLECTION=aist_chunks\n", encoding="utf-8"
    )
    assert Settings().qdrant_url == "http://localhost:6333"
