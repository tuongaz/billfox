"""Shared CLI helper functions for billfox commands."""

from __future__ import annotations

import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

import typer
from rich import print as rprint
from rich.markup import escape

# ── Config helpers ──────────────────────────────────────────────


def get_config_dir() -> Path:
    """Return the billfox config directory (~/.billfox)."""
    return Path.home() / ".billfox"


def get_config_file() -> Path:
    """Return the billfox config file path."""
    return get_config_dir() / "config.toml"


def read_config() -> dict[str, Any]:
    """Read config from ~/.billfox/config.toml."""
    config_file = get_config_file()
    if not config_file.exists():
        return {}
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def write_config(config: dict[str, Any]) -> None:
    """Write config to ~/.billfox/config.toml."""
    import tomli_w

    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    with open(config_file, "wb") as f:
        tomli_w.dump(config, f)


def get_nested(config: dict[str, Any], key: str) -> Any:
    """Get a nested value from config using dot-separated key."""
    parts = key.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested(config: dict[str, Any], key: str, value: str) -> None:
    """Set a nested value in config using dot-separated key."""
    parts = key.split(".")
    current: Any = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def flatten_config(
    config: dict[str, Any], prefix: str = "",
) -> list[tuple[str, str]]:
    """Flatten a nested config dict into (dotted_key, value) pairs."""
    items: list[tuple[str, str]] = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_config(value, full_key))
        else:
            items.append((full_key, str(value)))
    return items


# ── CLI guard ──────────────────────────────────────────────────


def ensure_configured() -> None:
    """Check that billfox has been configured via 'billfox init'.

    Exits with code 1 if config.toml is missing or defaults.ocr.provider is not set.
    """
    config = read_config()
    if get_nested(config, "defaults.ocr.provider") is None:

        rprint(
            "[yellow]billfox is not configured yet. "
            "Run 'billfox init' to set up.[/yellow]"
        )
        raise typer.Exit(code=1)


# ── Progress callbacks ─────────────────────────────────────────


def make_progress_callback() -> Any:
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
        elif event.status == Status.IN_PROGRESS:
            console.print(f"  [dim]{escape(event.message or '')}[/dim]", highlight=False)
        elif event.status == Status.COMPLETED:
            console.print(f"[bold blue]{stage_name}[/bold blue] [green]done[/green]", highlight=False)
        elif event.status == Status.FAILED:
            console.print(
                f"[bold blue]{stage_name}[/bold blue] [red]{escape(event.message or '')}[/red]",
                highlight=False,
            )

    return _on_progress


def make_step_callback() -> Any:
    """Create a synchronous step callback for extractor sub-steps, or None if not a TTY."""
    if not sys.stdout.isatty():
        return None

    from rich.console import Console

    console = Console(stderr=True)

    def _on_step(message: str) -> None:
        console.print(f"  [dim]{escape(message)}[/dim]", highlight=False)

    return _on_step


# ── Extractor / preprocessor builders ──────────────────────────


def build_preprocessors(preprocess: str | None, api_key: str | None = None) -> list[Any]:
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
            from billfox.preprocess.yolo import YOLOPreprocessor

            preprocessors.append(YOLOPreprocessor())
        else:
            raise typer.BadParameter(
                f"Unknown preprocessor: {name!r}. Available: resize, yolo"
            )
    return preprocessors


def build_extractor(extractor: str, api_key: str | None) -> Any:
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


# ── LLM model resolution ──────────────────────────────────────


def resolve_llm_model(
    model: str | None, config: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve the LLM model identifier and base_url from CLI arg and config.

    Returns (resolved_model, base_url).
    """
    base_url: str | None = None
    resolved_model = model

    if resolved_model is None:
        llm_provider = get_nested(config, "defaults.llm.provider")
        if llm_provider == "ollama":
            ollama_model = get_nested(config, "defaults.ollama.model")
            if ollama_model:
                resolved_model = f"ollama:{ollama_model}"
            base_url = get_nested(config, "defaults.ollama.base_url")
        else:
            config_model = get_nested(config, "defaults.llm.model")
            if config_model:
                resolved_model = config_model

    if resolved_model is None:
        resolved_model = "openai:gpt-4.1"

    # For any ollama: model (including CLI override), pick up base_url from config
    if resolved_model.startswith("ollama:") and base_url is None:
        base_url = get_nested(config, "defaults.ollama.base_url")

    return resolved_model, base_url


# ── Timezone helpers ──────────────────────────────────────────


def get_machine_timezone() -> str | None:
    """Detect the machine's IANA timezone, or None if unavailable."""
    try:
        tz = datetime.now().astimezone().tzinfo
        if tz is not None and hasattr(tz, "key"):
            key = tz.key  # type: ignore[union-attr]
            if key in available_timezones():
                return key  # type: ignore[no-any-return]
    except Exception:
        pass
    return None


def resolve_timezone_offset(
    expense_date: datetime | None,
    config: dict[str, Any],
) -> datetime | None:
    """Apply timezone to a naive expense_date using the fallback chain.

    If expense_date already has tzinfo, return as-is.
    Fallback: config defaults.timezone -> machine timezone.
    """
    if expense_date is None:
        return None
    if expense_date.tzinfo is not None:
        return expense_date

    # Try config timezone
    tz_name = get_nested(config, "defaults.timezone")
    if isinstance(tz_name, str) and tz_name in available_timezones():
        return expense_date.replace(tzinfo=ZoneInfo(tz_name))

    # Try machine timezone
    machine_tz = get_machine_timezone()
    if machine_tz:
        return expense_date.replace(tzinfo=ZoneInfo(machine_tz))

    return expense_date


# ── Search helpers ─────────────────────────────────────────────


def try_build_embedder() -> Any:
    """Try to build an embedder from config (Ollama or OpenAI)."""
    import os

    config = read_config()
    embedding_provider = get_nested(config, "defaults.embedding.provider")

    if embedding_provider == "ollama":
        embedding_model = get_nested(config, "defaults.embedding.model")
        if not embedding_model:
            return None
        base_url = get_nested(config, "defaults.ollama.base_url")
        from billfox.embed.ollama import OllamaEmbedder

        return OllamaEmbedder(model=embedding_model, base_url=base_url)

    if embedding_provider == "none":
        return None

    # Default / "openai" provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key_val = get_nested(config, "api_keys.openai")
        if isinstance(api_key_val, str):
            api_key = api_key_val
    if api_key:
        try:
            from billfox.embed.openai import OpenAIEmbedder

            return OpenAIEmbedder(api_key=api_key)
        except ImportError:
            return None
    return None
