"""Billfox CLI application."""

import asyncio
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


def _lazy_typer() -> Any:
    """Lazily import typer with clear error message."""
    try:
        import typer
    except ImportError:
        raise ImportError(
            "typer is required for the billfox CLI. "
            "Install it with: pip install 'billfox[cli]'"
        ) from None
    return typer


def _lazy_rich_print() -> Any:
    """Lazily import rich.print with clear error message."""
    try:
        from rich import print as rprint
    except ImportError:
        raise ImportError(
            "rich is required for the billfox CLI. "
            "Install it with: pip install 'billfox[cli]'"
        ) from None
    return rprint


typer = _lazy_typer()
app: Any = typer.Typer(
    name="billfox",
    help="Composable document data extraction: load, preprocess, OCR, LLM parse, store with vector search.",
    no_args_is_help=True,
)


def _make_progress_callback() -> Any:
    """Create a progress callback for CLI commands, or None if not a TTY."""
    if not sys.stdout.isatty():
        return None

    from rich.console import Console

    from billfox._progress import ProgressEvent, Status

    console = Console(stderr=True)

    async def _on_progress(event: ProgressEvent) -> None:
        stage_name = event.stage.value.lower()
        if event.status == Status.STARTED:
            console.print(f"[bold blue]{stage_name}[/bold blue]...", highlight=False)
        elif event.status == Status.COMPLETED:
            console.print(f"[bold blue]{stage_name}[/bold blue] [green]done[/green]", highlight=False)
        elif event.status == Status.FAILED:
            console.print(
                f"[bold blue]{stage_name}[/bold blue] [red]{event.message}[/red]",
                highlight=False,
            )

    return _on_progress


def _build_preprocessors(preprocess: str | None, api_key: str | None = None) -> list[Any]:
    """Build preprocessor list from comma-separated string."""
    if not preprocess:
        return []

    preprocessors: list[Any] = []
    for name in preprocess.split(","):
        name = name.strip().lower()
        if name == "resize":
            from billfox.preprocess.resize import ResizePreprocessor

            preprocessors.append(ResizePreprocessor())
        elif name == "yolo":
            raise typer.BadParameter(
                "YOLO preprocessor requires a model_path. "
                "Use the Python API directly for YOLO preprocessing."
            )
        else:
            raise typer.BadParameter(
                f"Unknown preprocessor: {name!r}. Available: resize, yolo"
            )
    return preprocessors


def _build_extractor(extractor: str, api_key: str | None) -> Any:
    """Build an extractor by name."""
    if extractor == "docling":
        from billfox.extract.docling import DoclingExtractor

        return DoclingExtractor()
    elif extractor == "mistral":
        from billfox.extract.mistral import MistralExtractor

        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        return MistralExtractor(**kwargs)
    else:
        raise typer.BadParameter(
            f"Unknown extractor: {extractor!r}. Available: docling, mistral"
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
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    on_progress = _make_progress_callback()

    async def _run() -> str:
        from billfox._progress import ProgressEvent, Stage, Status
        from billfox.source.local import LocalFileSource

        async def _emit(stage: Stage, status: Status, **kwargs: Any) -> None:
            if on_progress is not None:
                await on_progress(ProgressEvent(stage=stage, status=status, **kwargs))

        source = LocalFileSource()
        ext = _build_extractor(extractor, api_key)
        preprocessors = _build_preprocessors(preprocess)

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
            result = await ext.extract(document)
            await _emit(Stage.EXTRACTING, Status.COMPLETED, metadata={"pages": len(result.pages)})
        except Exception as e:
            await _emit(Stage.EXTRACTING, Status.FAILED, message=str(e))
            raise

        return str(result.markdown)

    try:
        markdown = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        rprint = _lazy_rich_print()
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    if output:
        Path(output).write_text(markdown, encoding="utf-8")
        rprint = _lazy_rich_print()
        rprint(f"[green]Output written to {output}[/green]")
    else:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")


@app.command()  # type: ignore[untyped-decorator]
def parse(
    file: str = typer.Argument(..., help="Path to the document file to parse."),
    schema: str = typer.Option(..., "--schema", "-s", help="Pydantic model in 'path:ClassName' format."),
    model: str = typer.Option("openai:gpt-4.1", "--model", "-m", help="LLM model identifier."),
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
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    schema_cls = _load_schema(schema)
    on_progress = _make_progress_callback()

    async def _run() -> Any:
        from billfox.parse.llm import LLMParser
        from billfox.pipeline import Pipeline
        from billfox.source.local import LocalFileSource

        source = LocalFileSource()
        ext = _build_extractor(extractor, api_key)
        preprocessors = _build_preprocessors(preprocess)
        parser: Any = LLMParser(
            model=model,
            output_type=schema_cls,
            system_prompt=prompt,
        )

        store_instance = None
        if store:
            from billfox.store.sqlite import SQLiteDocumentStore

            store_instance = SQLiteDocumentStore(
                db_path=store,
                schema=schema_cls,
            )

        pipeline: Any = Pipeline(
            source=source,
            extractor=ext,
            parser=parser,
            preprocessors=preprocessors,
            store=store_instance,
            on_progress=on_progress,
        )

        doc_id = Path(file).stem if store else None
        return await pipeline.run(file, document_id=doc_id)

    try:
        result = asyncio.run(_run())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        rprint = _lazy_rich_print()
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    output_text = (
        result.model_dump_json(indent=2) if json_output
        else json.dumps(result.model_dump(), indent=2)
    )

    if output:
        Path(output).write_text(output_text, encoding="utf-8")
        rprint = _lazy_rich_print()
        rprint(f"[green]Output written to {output}[/green]")
    else:
        sys.stdout.write(output_text)
        sys.stdout.write("\n")


# ── Config helpers ──────────────────────────────────────────────


def _get_config_dir() -> Path:
    """Return the billfox config directory (~/.billfox)."""
    return Path.home() / ".billfox"


def _get_config_file() -> Path:
    """Return the billfox config file path."""
    return _get_config_dir() / "config.toml"


def _read_config() -> dict[str, Any]:
    """Read config from ~/.billfox/config.toml."""
    config_file = _get_config_file()
    if not config_file.exists():
        return {}
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def _lazy_tomli_w() -> Any:
    """Lazily import tomli_w with clear error message."""
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "tomli-w is required for writing configuration. "
            "Install it with: pip install 'billfox[cli]'"
        ) from None
    return tomli_w


def _write_config(config: dict[str, Any]) -> None:
    """Write config to ~/.billfox/config.toml."""
    tw = _lazy_tomli_w()
    config_dir = _get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    with open(config_file, "wb") as f:
        tw.dump(config, f)


def _get_nested(config: dict[str, Any], key: str) -> Any:
    """Get a nested value from config using dot-separated key."""
    parts = key.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(config: dict[str, Any], key: str, value: str) -> None:
    """Set a nested value in config using dot-separated key."""
    parts = key.split(".")
    current: Any = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _flatten_config(
    config: dict[str, Any], prefix: str = "",
) -> list[tuple[str, str]]:
    """Flatten a nested config dict into (dotted_key, value) pairs."""
    items: list[tuple[str, str]] = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_config(value, full_key))
        else:
            items.append((full_key, str(value)))
    return items


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

app.command("backup")(backup_command)


@config_app.command("set")  # type: ignore[untyped-decorator]
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. api_keys.mistral)."),
    value: str = typer.Argument(..., help="Config value."),
) -> None:
    """Set a configuration value."""
    config = _read_config()
    _set_nested(config, key, value)
    _write_config(config)
    rprint = _lazy_rich_print()
    rprint(f"[green]Set {key} = {value}[/green]")


@config_app.command("get")  # type: ignore[untyped-decorator]
def config_get(
    key: str = typer.Argument(..., help="Config key (e.g. api_keys.mistral)."),
) -> None:
    """Get a configuration value."""
    config = _read_config()
    val = _get_nested(config, key)
    if val is None:
        rprint = _lazy_rich_print()
        rprint(f"[yellow]Key {key!r} not set[/yellow]")
        raise typer.Exit(code=1)
    typer.echo(val)


@config_app.command("list")  # type: ignore[untyped-decorator]
def config_list() -> None:
    """List all configuration values."""
    config = _read_config()
    if not config:
        rprint = _lazy_rich_print()
        rprint("[yellow]No configuration set.[/yellow]")
        return
    rprint = _lazy_rich_print()
    for k, v in _flatten_config(config):
        rprint(f"[bold]{k}[/bold] = {v}")


# ── Search helpers ──────────────────────────────────────────────


def _try_build_embedder() -> Any:
    """Try to build an OpenAI embedder from env or config."""
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        config = _read_config()
        api_key_val = _get_nested(config, "api_keys.openai")
        if isinstance(api_key_val, str):
            api_key = api_key_val
    if api_key:
        try:
            from billfox.embed.openai import OpenAIEmbedder

            return OpenAIEmbedder(api_key=api_key)
        except ImportError:
            return None
    return None


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

        embedder = _try_build_embedder()

        store_instance: Any = SQLiteDocumentStore(
            db_path=db,
            schema=_AnyModel,
            embedder=embedder,
        )
        return await store_instance.search(query, limit=limit, mode=mode)  # type: ignore[no-any-return]

    try:
        results = asyncio.run(_run())
    except Exception as e:
        rprint = _lazy_rich_print()
        rprint(f"[red]Error:[/red] {e}")
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
