"""Billfox CLI application."""

import asyncio
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

# ── Receipt sub-app ─────────────────────────────────────────

from billfox.cli.receipt import receipt_app  # noqa: E402

app.add_typer(receipt_app)

# ── Init command ─────────────────────────────────────────────

from billfox.cli.init import init as init_command  # noqa: E402

app.command("init")(init_command)

# ── llms.txt command ──────────────────────────────────────────

from billfox.cli.llms_txt import llms_txt as llms_txt_command  # noqa: E402

app.command("llms.txt")(llms_txt_command)


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


