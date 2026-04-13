"""CLI command for receipt parsing."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.markup import escape

import billfox.cli._helpers as _helpers


def receipt(
    file: str = typer.Argument(..., help="Path to the receipt file to parse."),
    model: str | None = typer.Option(
        None, "--model", "-m", help="LLM model identifier (reads from config if not set).",
    ),
    extractor: str = typer.Option("docling", "--extractor", "-e", help="Extractor to use (docling, mistral)."),
    preprocess: str | None = typer.Option(
        None, "--preprocess", "-p", help="Comma-separated preprocessors (e.g. resize).",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for the extractor."),
    store: str | None = typer.Option(
        None, "--store", help="SQLite database path (defaults to ~/.billfox/receipts.db).",
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file path. Defaults to stdout.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Parse a receipt into structured data using OCR + LLM."""
    import click as _click

    _ctx = _click.get_current_context()
    _has_override = (
        _ctx.get_parameter_source("model") == _click.core.ParameterSource.COMMANDLINE
        or _ctx.get_parameter_source("extractor") == _click.core.ParameterSource.COMMANDLINE
    )
    if not _has_override:
        _helpers.ensure_configured()

    # Use configured OCR provider when --extractor not passed
    if _ctx.get_parameter_source("extractor") != _click.core.ParameterSource.COMMANDLINE:
        config_ocr = _helpers.get_nested(_helpers.read_config(), "defaults.ocr.provider")
        if config_ocr:
            extractor = config_ocr

    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    # Resolve model and base_url from config when --model is not passed
    config = _helpers.read_config()
    resolved_model, base_url = _helpers.resolve_llm_model(model, config)

    on_progress = _helpers.make_progress_callback()
    on_step = _helpers.make_step_callback()

    # Default store to ~/.billfox/receipts.db
    db_path = store if store is not None else str(_helpers.get_config_dir() / "receipts.db")

    async def _run() -> Any:
        from billfox.models.prompts import RECEIPT_SYSTEM_PROMPT
        from billfox.models.receipt import Receipt
        from billfox.parse.llm import LLMParser
        from billfox.pipeline import Pipeline
        from billfox.source.local import LocalFileSource

        source = LocalFileSource()
        ext = _helpers.build_extractor(extractor, api_key)
        preprocessors = _helpers.build_preprocessors(preprocess)
        parser: Any = LLMParser(
            model=resolved_model,
            output_type=Receipt,
            system_prompt=RECEIPT_SYSTEM_PROMPT,
            base_url=base_url,
        )

        from billfox.store.sqlite import SQLiteDocumentStore

        store_instance: Any = SQLiteDocumentStore(
            db_path=db_path,
            schema=Receipt,
        )

        # Set up backup from config when storing
        backup_instance = None
        backup_provider = _helpers.get_nested(config, "defaults.backup.provider")
        backup_local_path = _helpers.get_nested(config, "defaults.backup.local_path")
        if backup_provider:
            from billfox.cli.backup import build_backup_from_config

            backup_instance = build_backup_from_config(
                backup_provider, backup_local_path,
            )

        pipeline: Any = Pipeline(
            source=source,
            extractor=ext,
            parser=parser,
            preprocessors=preprocessors,
            store=store_instance,
            backup=backup_instance,
            on_progress=on_progress,
            on_step=on_step,
        )

        doc_id = Path(file).stem
        return await pipeline.run(file, document_id=doc_id)

    try:
        result = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:

        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    output_text = (
        result.model_dump_json(indent=2) if json_output
        else json.dumps(result.model_dump(), indent=2)
    )

    if output:
        Path(output).write_text(output_text, encoding="utf-8")

        rprint(f"[green]Output written to {output}[/green]")
    else:
        sys.stdout.write(output_text)
        sys.stdout.write("\n")
