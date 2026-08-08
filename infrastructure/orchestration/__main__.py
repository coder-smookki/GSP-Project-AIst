from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aist-orchestration",
        description="Индексация корпуса через Prefect-поток",
    )
    parser.add_argument(
        "--source",
        choices=("api", "files"),
        default="files",
        help="источник корпуса (по умолчанию файлы)",
    )
    parser.add_argument(
        "--corpus-dir", default=None, help="каталог корпуса для --source files"
    )
    parser.add_argument(
        "--section", type=int, default=None, help="раздел для --source api"
    )
    parser.add_argument("--collection", default=None, help="коллекция-поколение")
    parser.add_argument("--generation", type=int, default=1, help="номер поколения")
    parser.add_argument("--limit", type=int, default=None, help="обработать не более N")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--switch-alias", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


async def run(args: argparse.Namespace) -> None:
    from infrastructure.orchestration.ingest_flow import ingest_flow

    await ingest_flow(
        source_kind=args.source,
        corpus_dir=args.corpus_dir,
        section_id=args.section,
        collection=args.collection,
        generation=args.generation,
        limit=args.limit,
        dry_run=args.dry_run,
        full=args.full,
        switch_alias=args.switch_alias,
    )


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    try:
        asyncio.run(run(args))
        return 0
    except KeyboardInterrupt:
        print("Прервано", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
