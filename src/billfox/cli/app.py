"""Billfox CLI application."""

import asyncio
import importlib.util
import json
import sys
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
    if extractor == "mistral":
        from billfox.extract.mistral import MistralExtractor

        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        return MistralExtractor(**kwargs)
    else:
        raise typer.BadParameter(
            f"Unknown extractor: {extractor!r}. Available: mistral"
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
    extractor: str = typer.Option("mistral", "--extractor", "-e", help="Extractor to use."),
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

    async def _run() -> str:
        from billfox.source.local import LocalFileSource

        source = LocalFileSource()
        ext = _build_extractor(extractor, api_key)
        preprocessors = _build_preprocessors(preprocess)

        document = await source.load(file)
        for pp in preprocessors:
            document = await pp.process(document)
        result = await ext.extract(document)
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
    extractor: str = typer.Option("mistral", "--extractor", "-e", help="Extractor to use."),
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
