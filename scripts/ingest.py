from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.logging import RichHandler

from infrastructure.config.settings import get_settings
from infrastructure.ingest.areas import area_title
from infrastructure.ingest.pipeline import (
    format_report,
    ingest_documents,
)
from infrastructure.retrieval.kb_client import KnowledgeBaseConfig
from infrastructure.retrieval.source import ApiSource, FilesSource

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILED = 2

console = Console()
logger = logging.getLogger("aist.ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Индексация корпуса базы знаний в Qdrant",
    )
    parser.add_argument(
        "--source",
        choices=("api", "files"),
        default="api",
        help="источник корпуса: API базы знаний или локальный каталог",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="каталог корпуса для --source files (по умолчанию из настроек)",
    )
    parser.add_argument(
        "--section",
        type=int,
        default=None,
        help="идентификатор раздела базы знаний (по умолчанию из настроек)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="обработать не более N документов"
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=1,
        help="номер поколения коллекции: данные пойдут в <алиас>_v<N>",
    )
    parser.add_argument(
        "--switch-alias",
        action="store_true",
        help="после успешного прогона переключить алиас на новое поколение",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="полная переиндексация: не пропускать неизменившиеся документы",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="разобрать корпус и посчитать чанки, ничего не записывая",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="подробный журнал работы"
    )
    return parser


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def build_source(args: argparse.Namespace, settings):
    if args.source == "files":
        return FilesSource(args.corpus_dir or settings.corpus_dir)

    if not settings.kb_email or not settings.kb_password:
        raise ValueError(
            "не заданы учётные данные базы знаний: задайте KB_EMAIL и KB_PASSWORD "
            "в окружении или в .env"
        )
    config = KnowledgeBaseConfig(
        base_url=settings.kb_base_url,
        email=settings.kb_email,
        password=settings.kb_password,
        section_id=args.section or settings.kb_section_id,
        token_name=settings.kb_token_name,
        per_page=settings.kb_per_page,
        max_retries=settings.kb_max_retries,
    )
    return ApiSource(config, offline=settings.aist_offline)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()

    try:
        source = build_source(args, settings)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Не удалось создать источник:[/red] {exc}")
        return EXIT_FAILED

    store = None
    embedder = None
    client = None
    collection = f"{settings.qdrant_collection}_v{args.generation}"

    if not args.dry_run:
        try:
            from qdrant_client import AsyncQdrantClient

            from infrastructure.config.model_registry import (
                get_embedding,
                setup_registry,
            )
            from infrastructure.retrieval.store import QdrantStore

            setup_registry(settings)
            embedder = get_embedding("default")
            client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=int(settings.qdrant_timeout),
            )
            store = QdrantStore(
                client,
                alias=settings.qdrant_collection,
                batch_size=settings.ingest_batch_size,
            )
            dim = await asyncio.to_thread(lambda: embedder.embedding_dim)
            await store.create_generation(collection, dim)
        except Exception as exc:
            console.print(f"[red]Индекс недоступен:[/red] {exc}")
            logger.exception("подготовка индекса не удалась")
            if client is not None:
                await client.close()
            return EXIT_FAILED

    sync_tag = f"run-{int(time.time())}"

    known_hashes: dict[str, str] | None = None
    if store is not None and client is not None and not args.full:
        from infrastructure.ingest.incremental import load_index_state

        known_hashes = await load_index_state(client, collection)

    def show(title: str, number: int, total: int) -> None:
        console.print(f"[dim]{number:>3}/{total}[/dim] {title[:70]}")

    try:
        await source.open()
        report = await ingest_documents(
            source,
            store,
            embedder,
            collection=collection,
            sync_tag=sync_tag,
            limit=args.limit,
            dry_run=args.dry_run,
            progress=show,
            known_hashes=known_hashes,
        )

        if (
            args.switch_alias
            and not args.dry_run
            and not report.failed
            and store is not None
        ):
            report.switched_from = await store.switch_generation(collection)
        elif args.switch_alias and report.failed:
            console.print(
                "[yellow]Алиас не переключён:[/yellow] в прогоне были ошибки, "
                "поколение не принято"
            )
    finally:
        await source.close()
        if client is not None:
            await client.close()

    console.print()
    console.print(format_report(report, area_title))

    return EXIT_PARTIAL if report.failed else EXIT_OK


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        console.print("[yellow]Прервано пользователем[/yellow]")
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
