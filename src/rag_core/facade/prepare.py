from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.file_io import detect_local_mime_type, read_file_bytes
from rag_core.core_models import Config, ParsedDocument, PreparedDocument
from rag_core._engine.core_prepare import parse_document_bytes, prepare_document_bytes

if TYPE_CHECKING:
    from rag_core.documents.contextualizer import ChunkContextualizer
    from rag_core.documents.ocr import OcrProvider
    from rag_core.events.sink import EventSink
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache


class _EnginePrepareMethods:
    _config: Config
    _ocr: "OcrProvider | None"
    _event_sink: "EventSink | None"
    _chunk_contextualizer: "ChunkContextualizer | None"
    _chunk_context_cache: "ChunkContextCache | None"

    async def parse_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> ParsedDocument:
        return await parse_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            event_sink=self._event_sink,
        )

    async def prepare_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
        namespace: str = "",
        collection: str = "",
        document_id: str = "",
    ) -> PreparedDocument:
        return await prepare_document_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            path=path,
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            ocr_provider=self._ocr,
            event_sink=self._event_sink,
            contextualizer=self._chunk_contextualizer,
            chunk_context_cache=self._chunk_context_cache,
            chunking_config=self._config.chunking,
        )

    async def prepare_file(
        self,
        path: str | Path,
        *,
        mime_type: str | None = None,
    ) -> PreparedDocument:
        file_path = Path(path)
        return await self.prepare_bytes(
            file_bytes=await read_file_bytes(file_path),
            filename=file_path.name,
            mime_type=mime_type or detect_local_mime_type(file_path),
            path=str(file_path),
        )
