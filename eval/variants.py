"""Сборка варианта: из оверрайдов настроек в поиск и агента.

Бенчмарк сравнивает варианты (модель, реранкер, ``top_k`` и т.д.), поэтому
сборка обязана работать не с глобальными настройками, а с переопределёнными
для конкретного варианта. ``Settings(**variant.settings)`` читает окружение и
применяет оверрайды сверху — один процесс может собрать и прогнать несколько
вариантов подряд.

Зависимости инфраструктуры импортируются внутри функций (DIP): модуль
``eval`` остаётся импортируемым и тестируемым без живых моделей.
"""

from __future__ import annotations

from eval.config import Environment, RunConfig, Variant, snapshot_environment
from eval.runners import BuildResult


def make_settings(variant: Variant):
    """Строит настройки варианта из оверрайдов.

    Args:
        variant: Вариант с оверрайдами настроек.

    Returns:
        Настройки приложения с применёнными оверрайдами.
    """
    from infrastructure.config.settings import Settings

    return Settings(**variant.settings)


def variant_environment(variant: Variant) -> Environment:
    """Снимает отпечаток окружения варианта без сборки зависимостей.

    Загрузка моделей не выполняется, поэтому можно посчитать ключ кэша до
    дорогого построения поиска/агента и пропустить прогон, если результат
    уже сохранён.

    Args:
        variant: Вариант системы.

    Returns:
        Environment: Отпечаток окружения варианта.
    """
    return snapshot_environment(make_settings(variant))


def build_search(settings):
    """Собирает поиск по базе знаний для настроек.

    Args:
        settings: Настройки варианта.

    Returns:
        Поиск по базе знаний.
    """
    from infrastructure.config.model_registry import setup_registry
    from infrastructure.retrieval.factory import get_search

    setup_registry(settings)
    return get_search(settings)


def build_agent(settings):
    """Собирает граф агента для настроек.

    Args:
        settings: Настройки варианта.

    Returns:
        Скомпилированный граф агента.
    """
    from infrastructure.config.composition_root import create_application

    return create_application(settings)


def build_variant(variant: Variant, config: RunConfig) -> BuildResult:
    """Собирает вариант в зависимости от прогона.

    Операционный замер требует агента; золотой набор — поиска, а агент — только
    если запрошены ответы. Несобранное не строится: загрузка модели в поиске
    без ответов стоит заметного времени.

    Args:
        variant: Вариант системы.
        config: Параметры прогона (определяют, что собирать).

    Returns:
        BuildResult: Поиск, агент (по требованию) и отпечаток окружения.
    """
    settings = make_settings(variant)
    environment = snapshot_environment(settings)

    if config.runner == "operational":
        return BuildResult(agent=build_agent(settings), environment=environment)

    search = build_search(settings)
    agent = build_agent(settings) if config.with_answers else None
    return BuildResult(search=search, agent=agent, environment=environment)
