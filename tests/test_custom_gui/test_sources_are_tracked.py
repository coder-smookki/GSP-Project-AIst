"""Исходники интерфейса попадают в репозиторий.

Дефект, ради которого написан этот тест: в ``.gitignore`` стоял питоновский
шаблон ``lib/`` без косой черты в начале, а такой шаблон совпадает с ЛЮБЫМ
каталогом ``lib`` на любой глубине. Весь ``custom_gui/web/src/lib/`` молча не
попал в репозиторий: локально сборка проходила, а на чужой машине падала с
``Could not resolve './lib/Login.svelte'``.

Ошибку такого рода не видит ни линтер, ни типизация, ни тесты приложения -
её видно только на чистом клоне. Поэтому проверка отдельная.

В ``.gitignore`` остались и другие шаблоны того же вида (``build/``, ``var/``,
``parts/``), поэтому проверяются все исходники, а не один каталог.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WEB_SRC = ROOT / "custom_gui" / "web" / "src"

#: Расширения, без которых сборка не соберётся.
SOURCE_SUFFIXES = {".svelte", ".ts", ".css"}


def _tracked() -> set[str]:
    """Возвращает пути, известные git, относительно корня репозитория.

    Returns:
        set[str]: Отслеживаемые пути.
    """
    result = subprocess.run(
        ["git", "ls-files", "custom_gui/web/src"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


@pytest.mark.skipif(not WEB_SRC.is_dir(), reason="исходники интерфейса не выгружены")
def test_every_frontend_source_is_tracked() -> None:
    """Каждый файл исходников интерфейса отслеживается git.

    Иначе сборка образа падает у всех, кроме автора: у него файл лежит
    на диске, а в репозитории его нет.
    """
    tracked = _tracked()
    on_disk = {
        str(path.relative_to(ROOT))
        for path in WEB_SRC.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    }
    missing = sorted(on_disk - tracked)

    assert not missing, (
        "не попали в репозиторий: "
        + ", ".join(missing)
        + ". Проверьте .gitignore: шаблон без косой черты в начале совпадает "
        "с каталогом такого имени на любой глубине"
    )


def test_ignore_patterns_for_build_dirs_are_anchored() -> None:
    """Шаблоны каталогов сборки привязаны к корню репозитория.

    ``lib/`` уже отправил в небытие весь фронтенд. У ``build/``, ``var/`` и
    ``parts/`` ровно та же природа: имя достаточно обычное, чтобы встретиться
    внутри подпроекта.
    """
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    risky = {"lib/", "lib64/", "build/", "var/", "parts/", "share/"}
    unanchored = [line for line in lines if line.strip() in risky]

    assert not unanchored, (
        "эти шаблоны совпадают с каталогом на любой глубине, "
        "добавьте косую черту в начало: " + ", ".join(unanchored)
    )
