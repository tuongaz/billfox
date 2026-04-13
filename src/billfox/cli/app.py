"""Billfox CLI application."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich import print as rprint
from rich.markup import escape

import billfox.cli._helpers as _helpers

# Load .env files early, before any command runs.
# Global (~/.billfox/.env) first, then project-local (./.env).
# Existing env vars take precedence (override=False is the default).
load_dotenv(Path.home() / ".billfox" / ".env")
load_dotenv(Path(".env"))

app: Any = typer.Typer(
    name="billfox",
    help="Composable document data extraction: load, preprocess, OCR, LLM parse, store with vector search.",
    no_args_is_help=True,
)


def _load_schema(schema_path: str) -> type[Any]:
    """Dynamically load a Pydantic model from path:ClassName format."""
    if ":" not in schema_path:
        raise typer.BadParameter(
            f"Schema must be in 'path:ClassName' format, got: {schema_path!r}"
        )

    file_path, class_name = schema_path.rsplit(":", 1)
    path = Path(file_path).resolve()

    if not path.exists():
        raise typer.BadParameter(f"Schema file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("_billfox_schema", str(path))
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise typer.BadParameter(
            f"Class {class_name!r} not found in {file_path}"
        )

    return getattr(module, class_name)  # type: ignore[no-any-return]


@app.command()  # type: ignore[untyped-decorator]
def extract(
    file: str = typer.Argument(..., help="Path to the document file to extract."),
    extractor: str = typer.Option("docling", "--extractor", "-e", help="Extractor to use (docling, mistral)."),
    preprocess: str | None = typer.Option(
        None, "--preprocess", "-p", help="Comma-separated preprocessors (e.g. resize).",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for the extractor."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path. Defaults to stdout."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Extract markdown from a document using OCR."""
    import click as _click

    _ctx = _click.get_current_context()
    if _ctx.get_parameter_source("extractor") != _click.core.ParameterSource.COMMANDLINE:
        _helpers.ensure_configured()
        config = _helpers.read_config()
        configured_provider = _helpers.get_nested(config, "defaults.ocr.provider")
        if configured_provider:
            extractor = configured_provider

    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    on_progress = _helpers.make_progress_callback()
    on_step = _helpers.make_step_callback()

    async def _run() -> str:
        from billfox._progress import ProgressEvent, Stage, Status
        from billfox.source.local import LocalFileSource

        async def _emit(stage: Stage, status: Status, **kwargs: Any) -> None:
            if on_progress is not None:
                await on_progress(ProgressEvent(stage=stage, status=status, **kwargs))

        source = LocalFileSource()
        ext = _helpers.build_extractor(extractor, api_key)
        preprocessors = _helpers.build_preprocessors(preprocess)

        try:
            await _emit(Stage.LOADING, Status.STARTED)
            document = await source.load(file)
            await _emit(Stage.LOADING, Status.COMPLETED)
        except Exception as e:
            await _emit(Stage.LOADING, Status.FAILED, message=str(e))
            raise

        for pp in preprocessors:
            try:
                await _emit(Stage.PREPROCESSING, Status.STARTED, message=type(pp).__name__)
                document = await pp.process(document)
                await _emit(Stage.PREPROCESSING, Status.COMPLETED)
            except Exception as e:
                await _emit(Stage.PREPROCESSING, Status.FAILED, message=str(e))
                raise

        try:
            await _emit(Stage.EXTRACTING, Status.STARTED)
            result = await ext.extract(document, on_step=on_step)
            await _emit(Stage.EXTRACTING, Status.COMPLETED, metadata={"pages": len(result.pages)})
        except Exception as e:
            await _emit(Stage.EXTRACTING, Status.FAILED, message=str(e))
            raise

        return str(result.markdown)

    try:
        markdown = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:


        rprint(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    if output:
        Path(output).write_text(markdown, encoding="utf-8")

        rprint(f"[green]Output written to {output}[/green]")
    else:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")


@app.command()  # type: ignore[untyped-decorator]
def parse(
    file: str = typer.Argument(..., help="Path to the document file to parse."),
    schema: str = typer.Option(..., "--schema", "-s", help="Pydantic model in 'path:ClassName' format."),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM model identifier (reads from config if not set)."),
    prompt: str = typer.Option(
        "Extract structured data from this document.",
        "--prompt",
        help="System prompt for the LLM parser.",
    ),
    extractor: str = typer.Option("docling", "--extractor", "-e", help="Extractor to use (docling, mistral)."),
    preprocess: str | None = typer.Option(
        None, "--preprocess", "-p", help="Comma-separated preprocessors.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for the extractor."),
    store: str | None = typer.Option(None, "--store", help="SQLite database path for storing results."),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file path. Defaults to stdout.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Parse a document into structured data using OCR + LLM."""
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

    schema_cls = _load_schema(schema)
    on_progress = _helpers.make_progress_callback()
    on_step = _helpers.make_step_callback()

    async def _run() -> Any:
        from billfox.parse.llm import LLMParser
        from billfox.pipeline import Pipeline
        from billfox.source.local import LocalFileSource

        source = LocalFileSource()
        ext = _helpers.build_extractor(extractor, api_key)
        preprocessors = _helpers.build_preprocessors(preprocess)
        parser: Any = LLMParser(
            model=resolved_model,
            output_type=schema_cls,
            system_prompt=prompt,
            base_url=base_url,
        )

        store_instance = None
        backup_instance = None
        if store:
            from billfox.store.sqlite import SQLiteDocumentStore

            store_instance = SQLiteDocumentStore(
                db_path=store,
                schema=schema_cls,
            )

            # Also set up backup from config when storing
            config = _helpers.read_config()
            backup_provider = _helpers.get_nested(config, "defaults.backup.provider")
            backup_local_path = _helpers.get_nested(config, "defaults.backup.local_path")
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

        doc_id = Path(file).stem if store else None
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


# ── Config sub-app ──────────────────────────────────────────────

config_app: Any = typer.Typer(
    name="config",
    help="Manage billfox configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app)

# ── Auth sub-app ───────────────────────────────────────────────

from billfox.cli.auth import auth_app  # noqa: E402

app.add_typer(auth_app)

# ── Backup command ────────────────────────────────────────────

from billfox.cli.backup import backup as backup_command  # noqa: E402
from billfox.cli.backup import build_backup_from_config  # noqa: E402

app.command("backup")(backup_command)

# ── Receipt command ──────────────────────────────────────────

from billfox.cli.receipt import receipt as receipt_command  # noqa: E402

app.command("receipt")(receipt_command)

# ── Init command ─────────────────────────────────────────────

from billfox.cli.init import init as init_command  # noqa: E402

app.command("init")(init_command)


@config_app.command("set")  # type: ignore[untyped-decorator]
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. api_keys.mistral)."),
    value: str = typer.Argument(..., help="Config value."),
) -> None:
    """Set a configuration value."""
    config = _helpers.read_config()
    _helpers.set_nested(config, key, value)
    _helpers.write_config(config)
    rprint(f"[green]Set {key} = {value}[/green]")


@config_app.command("get")  # type: ignore[untyped-decorator]
def config_get(
    key: str = typer.Argument(..., help="Config key (e.g. api_keys.mistral)."),
) -> None:
    """Get a configuration value."""
    config = _helpers.read_config()
    val = _helpers.get_nested(config, key)
    if val is None:

        rprint(f"[yellow]Key {key!r} not set[/yellow]")
        raise typer.Exit(code=1)
    typer.echo(val)


@config_app.command("list")  # type: ignore[untyped-decorator]
def config_list() -> None:
    """List all configuration values."""
    config = _helpers.read_config()
    if not config:

        rprint("[yellow]No configuration set.[/yellow]")
        return
    for k, v in _helpers.flatten_config(config):
        rprint(f"[bold]{k}[/bold] = {v}")


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


# ── Search command ──────────────────────────────────────────────


@app.command()  # type: ignore[untyped-decorator]
def search(
    query: str = typer.Argument(..., help="Search query."),
    db: str = typer.Option(..., "--db", "-d", help="SQLite database path."),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of results."),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode: hybrid, vector, or bm25.",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output machine-readable JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output."),
) -> None:
    """Search stored documents."""
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    if mode not in ("hybrid", "vector", "bm25"):
        raise typer.BadParameter(
            f"Invalid mode: {mode!r}. Choose from: hybrid, vector, bm25"
        )

    async def _run() -> list[Any]:
        from pydantic import BaseModel as _BaseModel
        from pydantic import ConfigDict

        from billfox.store.sqlite import SQLiteDocumentStore

        class _AnyModel(_BaseModel):
            model_config = ConfigDict(extra="allow")

        embedder = _helpers.try_build_embedder()

        store_instance: Any = SQLiteDocumentStore(
            db_path=db,
            schema=_AnyModel,
            embedder=embedder,
        )
        return await store_instance.search(query, limit=limit, mode=mode)  # type: ignore[no-any-return]

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
