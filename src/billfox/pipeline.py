from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

from billfox._progress import ProgressCallback, ProgressEvent, Stage, Status
from billfox._types import Document, ExtractionResult
from billfox.backup._base import DocumentBackup
from billfox.extract._base import Extractor, StepCallback
from billfox.parse._base import Parser
from billfox.preprocess._base import Preprocessor
from billfox.source._base import DocumentSource
from billfox.store._base import DocumentStore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class Pipeline(Generic[T]):
    """Composable document processing pipeline.

    Wires together source -> preprocess -> extract -> parse -> store.
    """

    source: DocumentSource
    extractor: Extractor
    parser: Parser[T]
    preprocessors: list[Preprocessor] = field(default_factory=list)
    store: DocumentStore[T] | None = None
    backup: DocumentBackup | None = None
    on_progress: ProgressCallback | None = None
    on_step: StepCallback | None = None

    async def _emit(
        self,
        stage: Stage,
        status: Status,
        *,
        message: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.on_progress is None:
            return
        await self.on_progress(ProgressEvent(stage=stage, status=status, message=message, metadata=metadata))

    async def run(self, uri: str, document_id: str | None = None) -> T:
        """Execute the full pipeline: load -> preprocess -> extract -> parse -> store."""
        # LOADING
        try:
            await self._emit(Stage.LOADING, Status.STARTED)
            document = await self.source.load(uri)
            await self._emit(Stage.LOADING, Status.COMPLETED)
        except Exception as exc:
            await self._emit(Stage.LOADING, Status.FAILED, message=str(exc))
            raise

        # PREPROCESSING
        original_document: Document | None = None
        if self.preprocessors:
            try:
                for p in self.preprocessors:
                    await self._emit(Stage.PREPROCESSING, Status.STARTED, message=type(p).__name__)
                original_document = document
                document = await self._preprocess(document)
                await self._emit(Stage.PREPROCESSING, Status.COMPLETED)
            except Exception as exc:
                await self._emit(Stage.PREPROCESSING, Status.FAILED, message=str(exc))
                raise

        # EXTRACTING
        try:
            await self._emit(Stage.EXTRACTING, Status.STARTED)
            result = await self.extractor.extract(document, on_step=self.on_step)
            await self._emit(Stage.EXTRACTING, Status.COMPLETED, metadata={"pages": len(result.pages)})
        except Exception as exc:
            await self._emit(Stage.EXTRACTING, Status.FAILED, message=str(exc))
            raise

        # PARSING
        try:
            await self._emit(Stage.PARSING, Status.STARTED)
            parsed = await self.parser.parse(result.markdown)
            await self._emit(Stage.PARSING, Status.COMPLETED)
        except Exception as exc:
            await self._emit(Stage.PARSING, Status.FAILED, message=str(exc))
            raise

        # STORING
        if self.store is not None and document_id is not None:
            try:
                await self._emit(Stage.STORING, Status.STARTED)
                await self.store.save(document_id, parsed)
                await self._emit(Stage.STORING, Status.COMPLETED)
            except Exception as exc:
                await self._emit(Stage.STORING, Status.FAILED, message=str(exc))
                raise

        # BACKUP
        if self.backup is not None:
            try:
                backup_result = await self.backup.backup(document, original=original_document)
                if self.store is not None and document_id is not None and hasattr(self.store, "save_file_paths"):
                    await self.store.save_file_paths(
                        document_id,
                        file_path=backup_result.uri,
                        original_file_path=backup_result.original_uri,
                    )
            except Exception:
                logger.warning("Backup failed", exc_info=True)

        return parsed

    async def extract_only(self, uri: str) -> ExtractionResult:
        """Execute load -> preprocess -> extract only (no parse or store)."""
        # LOADING
        try:
            await self._emit(Stage.LOADING, Status.STARTED)
            document = await self.source.load(uri)
            await self._emit(Stage.LOADING, Status.COMPLETED)
        except Exception as exc:
            await self._emit(Stage.LOADING, Status.FAILED, message=str(exc))
            raise

        # PREPROCESSING
        original_document: Document | None = None
        if self.preprocessors:
            try:
                for p in self.preprocessors:
                    await self._emit(Stage.PREPROCESSING, Status.STARTED, message=type(p).__name__)
                original_document = document
                document = await self._preprocess(document)
                await self._emit(Stage.PREPROCESSING, Status.COMPLETED)
            except Exception as exc:
                await self._emit(Stage.PREPROCESSING, Status.FAILED, message=str(exc))
                raise

        # EXTRACTING
        try:
            await self._emit(Stage.EXTRACTING, Status.STARTED)
            result = await self.extractor.extract(document, on_step=self.on_step)
            await self._emit(Stage.EXTRACTING, Status.COMPLETED, metadata={"pages": len(result.pages)})
        except Exception as exc:
            await self._emit(Stage.EXTRACTING, Status.FAILED, message=str(exc))
            raise

        # BACKUP
        if self.backup is not None:
            try:
                await self.backup.backup(document, original=original_document)
            except Exception:
                logger.warning("Backup failed", exc_info=True)

        return result

    async def _preprocess(self, document: Document) -> Document:
        for preprocessor in self.preprocessors:
            document = await preprocessor.process(document)
        return document
