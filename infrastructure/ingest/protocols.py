from __future__ import annotations

from typing import Protocol, runtime_checkable

from infrastructure.ingest.models import IngestChunk, ParsedDocument
from infrastructure.retrieval.source import RawDocument


class DocumentParser(Protocol):
    kind: str

    def parse(self, raw: RawDocument) -> ParsedDocument: ...


class Deanonymizer(Protocol):
    def process(self, document: ParsedDocument) -> ParsedDocument: ...


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> list[IngestChunk]: ...


class Classifier(Protocol):
    def classify(self, doc_title: str) -> list[str]: ...


@runtime_checkable
class Embedder(Protocol):
    async def aembed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def create_generation(self, name: str, dim: int) -> None: ...

    async def upsert_chunks(
        self,
        collection: str,
        chunks: list[IngestChunk],
        vectors: list[list[float]],
        areas: list[str],
        sync_tag: str,
        content_hash: str,
    ) -> int: ...

    async def sweep_document(
        self, collection: str, doc_id: str, sync_tag: str
    ) -> None: ...

    async def switch_generation(self, collection: str) -> str | None: ...
