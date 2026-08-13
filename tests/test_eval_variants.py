"""Тесты сборки вариантов бенчмарка.

Сборка из оверрайдов — ключ к A/B: если оверрайды не применяются, оба варианта
прогоняются на одних настройках и сравнение бессмысленно. Тяжёлые зависимости
(модели, Qdrant) подменяются: поиск на заглушке не требует сети, а построение
агента в тестах не запускается без необходимости.
"""

from __future__ import annotations

from eval.config import RunConfig, Variant
from eval.variants import build_variant, make_settings

MOCK_OVERRIDES = {"retrieval_provider": "mock", "use_mock_search": True}


class TestMakeSettings:
    """Настройки варианта из оверрайдов."""

    def test_applies_overrides(self) -> None:
        """Оверрайды применяются поверх окружения."""
        settings = make_settings(Variant(name="x", settings={"retrieval_top_k": 6}))
        assert settings.retrieval_top_k == 6

    def test_empty_overrides_keep_defaults(self) -> None:
        """Пустые оверрайды дают базовые настройки."""
        settings = make_settings(Variant(name="base"))
        assert settings.retrieval_top_k == 5

    def test_mock_provider_forced(self) -> None:
        """Заглушка поиска включается оверрайдом."""
        settings = make_settings(Variant(name="x", settings=MOCK_OVERRIDES))
        assert settings.resolved_retrieval_provider() == "mock"


class TestBuildVariant:
    """Сборка варианта по типу прогона."""

    def test_golden_without_answers_builds_search_only(self) -> None:
        """Поиск без ответов строит поиск, а не агента."""
        variant = Variant(name="base", settings=MOCK_OVERRIDES)
        built = build_variant(variant, RunConfig(runner="golden", with_answers=False))
        assert built.search is not None
        assert built.agent is None
        assert built.environment is not None

    def test_golden_with_answers_builds_agent(self) -> None:
        """С ответами строится и агент."""
        variant = Variant(name="base", settings=MOCK_OVERRIDES)
        built = build_variant(variant, RunConfig(runner="golden", with_answers=True))
        assert built.search is not None
        assert built.agent is not None

    def test_operational_builds_agent(self) -> None:
        """Операционный замер требует агента."""
        variant = Variant(name="base", settings=MOCK_OVERRIDES)
        built = build_variant(variant, RunConfig(runner="operational"))
        assert built.agent is not None

    def test_environment_fingerprint_differs_by_model(self) -> None:
        """Смена модели в оверрайдах меняет отпечаток окружения."""
        base = build_variant(
            Variant(name="base", settings=MOCK_OVERRIDES),
            RunConfig(runner="golden"),
        )
        other = build_variant(
            Variant(name="new", settings={**MOCK_OVERRIDES, "llm_model": "other"}),
            RunConfig(runner="golden"),
        )
        assert base.environment.fingerprint() != other.environment.fingerprint()
