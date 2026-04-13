"""CLI sub-app for receipt operations: parse, search, list, get, delete, edit."""

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

RECEIPT_EMBED_FIELDS = ["search_text"]

receipt_app: Any = typer.Typer(
    name="receipt",
    help="Parse, search, list, delete and edit receipts.",
    no_args_is_help=True,
)


# ── parse ────────────────────────────────────────────────────────


@receipt_app.command()  # type: ignore[untyped-decorator]
def parse(
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
        # Always crop receipts with YOLO + resize when no --preprocess specified
        if not preprocessors:
            preprocessors = _helpers.build_preprocessors("yolo,resize")
        parser: Any = LLMParser(
            model=resolved_model,
            output_type=Receipt,
            system_prompt=RECEIPT_SYSTEM_PROMPT,
            base_url=base_url,
        )

        from billfox.store.sqlite import SQLiteDocumentStore

        embedder = _helpers.try_build_embedder()
        store_instance: Any = SQLiteDocumentStore(
            db_path=db_path,
            schema=Receipt,
            embedder=embedder,
            embed_fields=RECEIPT_EMBED_FIELDS,
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

        from billfox._id import generate_id

        doc_id = generate_id()
        try:
            parsed = await pipeline.run(file, document_id=doc_id)
            # Remove invalid receipts (no vendor and no total and no items)
            if not parsed.vendor_name and parsed.total is None and not parsed.items:
                await store_instance.delete(doc_id)
                return None
            return parsed
        finally:
            await store_instance.close()

    try:
        result = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:

        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    if result is None:
        rprint("[yellow]Document is not a valid receipt. Skipping.[/yellow]")
        raise typer.Exit(code=0)

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


# ── search ───────────────────────────────────────────────────────


def _display_search_results(results: list[Any]) -> None:
    """Display search results as a formatted rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        if not results:
            sys.stdout.write("No results found.\n")
            return
        for i, r in enumerate(results, 1):
            sys.stdout.write(
                f"{i}. {r.document_id} (score: {r.score:.4f})\n"
            )
        return

    console = Console()
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("Rank", style="dim")
    table.add_column("Document ID", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Data")

    for i, r in enumerate(results, 1):
        data_str = json.dumps(r.data, default=str)
        if len(data_str) > 80:
            data_str = data_str[:77] + "..."
        table.add_row(str(i), r.document_id, f"{r.score:.4f}", data_str)

    console.print(table)


@receipt_app.command()  # type: ignore[untyped-decorator]
def search(
    query: str = typer.Argument(..., help="Search query."),
    db: str = typer.Option(
        str(Path.home() / ".billfox" / "receipts.db"),
        "--db", "-d",
        help="SQLite database path.",
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of results."),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode: hybrid, vector, or bm25.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Search stored receipts."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    if mode not in ("hybrid", "vector", "bm25"):
        raise typer.BadParameter(
            f"Invalid mode: {mode!r}. Choose from: hybrid, vector, bm25"
        )

    async def _run() -> list[Any]:
        from billfox.models.receipt import Receipt
        from billfox.store.sqlite import SQLiteDocumentStore

        embedder = _helpers.try_build_embedder()

        store_instance: Any = SQLiteDocumentStore(
            db_path=db,
            schema=Receipt,
            embedder=embedder,
            embed_fields=RECEIPT_EMBED_FIELDS,
        )
        try:
            return await store_instance.search(query, limit=limit, mode=mode)  # type: ignore[no-any-return]
        finally:
            await store_instance.close()

    try:
        results = asyncio.run(_run())
    except Exception as e:

        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    if json_output:
        output_text = json.dumps(
            [
                {
                    "document_id": r.document_id,
                    "score": r.score,
                    "data": r.data,
                    "signals": dict(r.signals),
                }
                for r in results
            ],
            indent=2,
            default=str,
        )
        sys.stdout.write(output_text)
        sys.stdout.write("\n")
    else:
        _display_search_results(results)


# ── list ─────────────────────────────────────────────────────────


@receipt_app.command("list")  # type: ignore[untyped-decorator]
def list_receipts(
    db: str = typer.Option(
        str(Path.home() / ".billfox" / "receipts.db"),
        "--db", "-d",
        help="SQLite database path.",
    ),
    page: int = typer.Option(1, "--page", "-p", help="Page number (starts at 1)."),
    per_page: int = typer.Option(20, "--per-page", "-n", help="Items per page."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """List stored receipts with pagination."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    if page < 1:
        raise typer.BadParameter("Page must be >= 1.")

    offset = (page - 1) * per_page

    async def _run() -> tuple[list[tuple[str, Any]], int]:
        from billfox.models.receipt import Receipt
        from billfox.store.sqlite import SQLiteDocumentStore

        store_instance: Any = SQLiteDocumentStore(
            db_path=db,
            schema=Receipt,
        )
        try:
            return await store_instance.list_documents(limit=per_page, offset=offset)  # type: ignore[no-any-return]
        finally:
            await store_instance.close()

    try:
        items, total = asyncio.run(_run())
    except Exception as e:

        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    if json_output:
        output_text = json.dumps(
            {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "items": [
                    {"document_id": doc_id, "data": data.model_dump()}
                    for doc_id, data in items
                ],
            },
            indent=2,
            default=str,
        )
        sys.stdout.write(output_text)
        sys.stdout.write("\n")
    else:
        _display_list_results(items, page=page, total=total, total_pages=total_pages)


def _display_list_results(
    items: list[tuple[str, Any]],
    *,
    page: int,
    total: int,
    total_pages: int,
) -> None:
    """Display list results as a formatted rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        if not items:
            sys.stdout.write("No receipts found.\n")
            return
        for doc_id, data in items:
            sys.stdout.write(f"{doc_id}: {json.dumps(data.model_dump(), default=str)}\n")
        sys.stdout.write(f"\nPage {page}/{total_pages} ({total} total)\n")
        return

    console = Console()
    if not items:
        console.print("[yellow]No receipts found.[/yellow]")
        return

    table = Table(title=f"Receipts — page {page}/{total_pages} ({total} total)")
    table.add_column("#", style="dim")
    table.add_column("Document ID", style="bold")
    table.add_column("Vendor")
    table.add_column("Total", justify="right")
    table.add_column("Date")
    table.add_column("Currency")

    for idx, (doc_id, data) in enumerate(items, 1):
        d = data.model_dump()
        table.add_row(
            str(idx),
            doc_id,
            str(d.get("vendor_name") or ""),
            str(d.get("total") or ""),
            str(d.get("expense_date") or ""),
            str(d.get("currency") or ""),
        )

    console.print(table)


# ── get ──────────────────────────────────────────────────────────


@receipt_app.command("get")  # type: ignore[untyped-decorator]
def get_receipt(
    document_id: str = typer.Argument(..., help="Receipt document ID."),
    original: bool = typer.Option(False, "--original", "-o", help="Get the original (pre-crop) file instead."),
    db: str = typer.Option(
        str(Path.home() / ".billfox" / "receipts.db"),
        "--db", "-d",
        help="SQLite database path.",
    ),
) -> None:
    """Print the file path of a stored receipt's cropped or original file."""

    async def _run() -> tuple[str | None, str | None]:
        from billfox.models.receipt import Receipt
        from billfox.store.sqlite import SQLiteDocumentStore

        store_instance: Any = SQLiteDocumentStore(db_path=db, schema=Receipt)
        try:
            return await store_instance.get_file_paths(document_id)  # type: ignore[no-any-return]
        finally:
            await store_instance.close()

    try:
        file_path, original_file_path = asyncio.run(_run())
    except Exception as e:
        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    path = original_file_path if original else file_path
    if path is None:
        label = "original" if original else "cropped"
        rprint(f"[yellow]No {label} file path stored for {document_id!r}.[/yellow]")
        raise typer.Exit(code=1)

    sys.stdout.write(path)
    sys.stdout.write("\n")


# ── delete ────────────────────────────────────────────────────────


@receipt_app.command("delete")  # type: ignore[untyped-decorator]
def delete_receipt(
    document_id: str = typer.Argument(..., help="Receipt document ID to delete."),
    db: str = typer.Option(
        str(Path.home() / ".billfox" / "receipts.db"),
        "--db", "-d",
        help="SQLite database path.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Delete a stored receipt by document ID."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    async def _run() -> bool:
        from billfox.models.receipt import Receipt
        from billfox.store.sqlite import SQLiteDocumentStore

        store_instance: Any = SQLiteDocumentStore(db_path=db, schema=Receipt)
        try:
            existing = await store_instance.get(document_id)
            if existing is None:
                return False
            await store_instance.delete(document_id)
            return True
        finally:
            await store_instance.close()

    try:
        deleted = asyncio.run(_run())
    except Exception as e:
        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    if not deleted:
        if json_output:
            sys.stdout.write(json.dumps({"error": "not_found", "document_id": document_id}))
            sys.stdout.write("\n")
        else:
            rprint(f"[yellow]Receipt {document_id!r} not found.[/yellow]")
        raise typer.Exit(code=1)

    if json_output:
        sys.stdout.write(json.dumps({"deleted": True, "document_id": document_id}))
        sys.stdout.write("\n")
    else:
        rprint(f"[green]Deleted receipt {document_id!r}.[/green]")


# ── edit ──────────────────────────────────────────────────────────

@receipt_app.command("edit")  # type: ignore[untyped-decorator]
def edit_receipt(
    document_id: str = typer.Argument(..., help="Receipt document ID to edit."),
    data: str | None = typer.Option(None, "--data", help="JSON string with fields to update."),
    vendor_name: str | None = typer.Option(None, "--vendor-name", help="Vendor name."),
    total: float | None = typer.Option(None, "--total", help="Total amount."),
    expense_date: str | None = typer.Option(None, "--expense-date", help="Expense date."),
    currency: str | None = typer.Option(None, "--currency", help="Currency code."),
    tax_amount: float | None = typer.Option(None, "--tax-amount", help="Tax amount."),
    tax_rate: float | None = typer.Option(None, "--tax-rate", help="Tax rate."),
    payment_method: str | None = typer.Option(None, "--payment-method", help="Payment method."),
    invoice_number: str | None = typer.Option(None, "--invoice-number", help="Invoice number."),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags."),
    db: str = typer.Option(
        str(Path.home() / ".billfox" / "receipts.db"),
        "--db", "-d",
        help="SQLite database path.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Edit fields of a stored receipt."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    # Build updates from --data JSON
    updates: dict[str, Any] = {}
    if data is not None:
        try:
            updates = json.loads(data)
        except json.JSONDecodeError as e:
            rprint(f"[red]Error:[/red] Invalid JSON for --data: {escape(str(e))}")
            raise typer.Exit(code=1) from None
        if not isinstance(updates, dict):
            rprint("[red]Error:[/red] --data must be a JSON object.")
            raise typer.Exit(code=1)

    # Individual flags override --data values
    if vendor_name is not None:
        updates["vendor_name"] = vendor_name
    if total is not None:
        updates["total"] = total
    if expense_date is not None:
        updates["expense_date"] = expense_date
    if currency is not None:
        updates["currency"] = currency
    if tax_amount is not None:
        updates["tax_amount"] = tax_amount
    if tax_rate is not None:
        updates["tax_rate"] = tax_rate
    if payment_method is not None:
        updates["payment_method"] = payment_method
    if invoice_number is not None:
        updates["invoice_number"] = invoice_number
    if tags is not None:
        updates["tags"] = [t.strip() for t in tags.split(",")]

    if not updates:
        rprint("[yellow]No updates provided. Use --data or field flags (e.g. --vendor-name).[/yellow]")
        raise typer.Exit(code=1)

    async def _run() -> Any:
        from billfox.models.receipt import Receipt
        from billfox.store.sqlite import SQLiteDocumentStore

        embedder = _helpers.try_build_embedder()
        store_instance: Any = SQLiteDocumentStore(
            db_path=db,
            schema=Receipt,
            embedder=embedder,
            embed_fields=RECEIPT_EMBED_FIELDS,
        )
        try:
            existing = await store_instance.get(document_id)
            if existing is None:
                return None
            updated = existing.model_copy(update=updates)
            await store_instance.save(document_id, updated)
            return updated
        finally:
            await store_instance.close()

    try:
        result = asyncio.run(_run())
    except Exception as e:
        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    if result is None:
        if json_output:
            sys.stdout.write(json.dumps({"error": "not_found", "document_id": document_id}))
            sys.stdout.write("\n")
        else:
            rprint(f"[yellow]Receipt {document_id!r} not found.[/yellow]")
        raise typer.Exit(code=1)

    output_text = (
        result.model_dump_json(indent=2) if json_output
        else json.dumps(result.model_dump(), indent=2)
    )
    sys.stdout.write(output_text)
    sys.stdout.write("\n")
