"""Billfox init setup wizard."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import available_timezones

import httpx
import typer
from rich.console import Console

_COMMON_TIMEZONES = [
    ("Australia/Sydney", "Australia — Sydney (AEST/AEDT)"),
    ("Australia/Melbourne", "Australia — Melbourne (AEST/AEDT)"),
    ("Australia/Brisbane", "Australia — Brisbane (AEST)"),
    ("Australia/Perth", "Australia — Perth (AWST)"),
    ("Australia/Adelaide", "Australia — Adelaide (ACST/ACDT)"),
    ("Pacific/Auckland", "New Zealand (NZST/NZDT)"),
    ("Asia/Tokyo", "Japan (JST)"),
    ("Asia/Singapore", "Singapore (SGT)"),
    ("Asia/Shanghai", "China (CST)"),
    ("Europe/London", "UK (GMT/BST)"),
    ("Europe/Paris", "Central Europe (CET/CEST)"),
    ("America/New_York", "US — Eastern (EST/EDT)"),
    ("America/Chicago", "US — Central (CST/CDT)"),
    ("America/Denver", "US — Mountain (MST/MDT)"),
    ("America/Los_Angeles", "US — Pacific (PST/PDT)"),
]


def _check_ollama(base_url: str) -> list[str] | None:
    """Check Ollama connectivity and return list of model names, or None on failure."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _prompt_choice(prompt_text: str, choices: list[str], descriptions: list[str]) -> int:
    """Display numbered choices and return the 1-based selection index."""
    console = Console(stderr=True)

    console.print(f"\n[bold]{prompt_text}[/bold]")
    for i, (choice, desc) in enumerate(zip(choices, descriptions, strict=True), 1):
        if desc:
            console.print(f"  {i}) {choice} — {desc}")
        else:
            console.print(f"  {i}) {choice}")

    while True:
        raw = typer.prompt("Choose", default="1")
        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return idx
        except ValueError:
            pass
        console.print(f"[red]Please enter a number between 1 and {len(choices)}[/red]")


def init(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation when overwriting existing config."),
) -> None:
    """Interactive setup wizard for billfox."""
    from billfox.cli._helpers import get_config_dir, read_config, write_config

    console = Console(stderr=True)

    config_dir = get_config_dir()
    config_file = config_dir / "config.toml"

    # Check for existing config
    existing_config = read_config()
    if existing_config and not yes:
        console.print("\n[yellow]Existing configuration found.[/yellow]")
        overwrite = typer.confirm("Overwrite existing config?", default=False)
        if not overwrite:
            console.print("[dim]Setup cancelled.[/dim]")
            raise typer.Exit(code=0)

    console.print("\n[bold blue]Welcome to billfox setup![/bold blue]")
    console.print("This wizard will configure your OCR, LLM, and backup preferences.\n")

    # ── OCR Provider ──────────────────────────────────────────────
    ocr_choice = _prompt_choice(
        "Select OCR provider:",
        ["Docling", "Mistral"],
        ["local, free", "API, requires key"],
    )
    ocr_provider = "docling" if ocr_choice == 1 else "mistral"

    # ── LLM Provider ─────────────────────────────────────────────
    llm_choice = _prompt_choice(
        "Select LLM provider:",
        ["OpenAI", "Claude (Anthropic)", "Ollama"],
        ["API, requires key", "API, requires key", "local, no key needed"],
    )

    llm_provider: str
    llm_model: str
    ollama_base_url: str | None = None
    ollama_model: str | None = None

    if llm_choice == 1:
        llm_provider = "openai"
        llm_model = "openai:gpt-4.1"
    elif llm_choice == 2:
        llm_provider = "anthropic"
        llm_model = "anthropic:claude-sonnet-4-20250514"
    else:
        llm_provider = "ollama"
        ollama_base_url = typer.prompt(
            "Ollama base URL",
            default="http://localhost:11434",
        )

        console.print(f"\n[dim]Checking Ollama at {ollama_base_url}...[/dim]")
        available_models = _check_ollama(ollama_base_url)

        if available_models and len(available_models) > 0:
            console.print(f"[green]Connected! Found {len(available_models)} model(s):[/green]")
            model_idx = _prompt_choice(
                "Select a model:",
                available_models,
                [""] * len(available_models),
            )
            ollama_model_name = available_models[model_idx - 1]
        else:
            if available_models is not None:
                console.print("[yellow]Connected to Ollama but no models found.[/yellow]")
            else:
                console.print(
                    f"[yellow]Could not connect to Ollama at {ollama_base_url}.[/yellow]"
                )
            console.print("[dim]You can pull models later with: ollama pull <model>[/dim]")
            ollama_model_name = typer.prompt("Ollama model name", default="llama3.2")

        ollama_model = ollama_model_name
        llm_model = f"ollama:{ollama_model_name}"

    # ── Embedding Provider ───────────────────────────────────────
    embedding_choices = ["OpenAI", "Ollama", "None"]
    embedding_descs = ["API, requires key", "local, no key needed", "skip embeddings (no vector search)"]

    embedding_choice = _prompt_choice(
        "Select embedding provider (for vector search):",
        embedding_choices,
        embedding_descs,
    )

    embedding_provider: str
    embedding_model: str | None = None

    if embedding_choice == 1:
        embedding_provider = "openai"
        embedding_model = "text-embedding-3-small"
    elif embedding_choice == 2:
        embedding_provider = "ollama"
        # Reuse ollama_base_url if already set from LLM selection
        emb_base_url = ollama_base_url
        if emb_base_url is None:
            emb_base_url = typer.prompt(
                "Ollama base URL",
                default="http://localhost:11434",
            )
            ollama_base_url = emb_base_url

        console.print(f"\n[dim]Checking Ollama at {emb_base_url}...[/dim]")
        emb_available_models = _check_ollama(emb_base_url)

        if emb_available_models and len(emb_available_models) > 0:
            console.print(f"[green]Connected! Found {len(emb_available_models)} model(s):[/green]")
            emb_model_idx = _prompt_choice(
                "Select an embedding model:",
                emb_available_models,
                [""] * len(emb_available_models),
            )
            embedding_model = emb_available_models[emb_model_idx - 1]
        else:
            if emb_available_models is not None:
                console.print("[yellow]Connected to Ollama but no models found.[/yellow]")
            else:
                console.print(
                    f"[yellow]Could not connect to Ollama at {emb_base_url}.[/yellow]"
                )
            console.print("[dim]You can pull models later with: ollama pull <model>[/dim]")
            embedding_model = typer.prompt("Ollama embedding model name", default="nomic-embed-text")
    else:
        embedding_provider = "none"

    # ── Backup Provider ──────────────────────────────────────────
    backup_choice = _prompt_choice(
        "Select backup provider:",
        ["Local folder", "Google Drive"],
        ["saves to a local directory", "requires Google Drive auth"],
    )

    backup_provider: str
    backup_local_path: str | None = None

    if backup_choice == 1:
        backup_provider = "local"
        default_path = str(config_dir / "backups")
        backup_local_path = typer.prompt("Backup folder path", default=default_path)
        # Create directory if needed
        Path(backup_local_path).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created backup directory: {backup_local_path}[/green]")
    else:
        backup_provider = "google_drive"
        console.print(
            "\n[dim]To authenticate with Google Drive, run:[/dim] "
            "[bold]billfox auth google-drive[/bold]"
        )

    # ── Timezone ─────────────────────────────────────────────────
    from billfox.cli._helpers import get_machine_timezone

    detected_tz = get_machine_timezone()

    tz_choices: list[str] = []
    tz_descriptions: list[str] = []

    if detected_tz:
        tz_choices.append(detected_tz)
        tz_descriptions.append(f"detected from system")

    for tz_id, tz_label in _COMMON_TIMEZONES:
        if tz_id != detected_tz:
            tz_choices.append(tz_id)
            tz_descriptions.append(tz_label)

    tz_choices.append("Other")
    tz_descriptions.append("type IANA timezone manually")

    tz_idx = _prompt_choice("Select your timezone:", tz_choices, tz_descriptions)
    selected_tz: str

    if tz_choices[tz_idx - 1] == "Other":
        valid_tzs = available_timezones()
        while True:
            raw_tz = typer.prompt("IANA timezone (e.g. America/New_York)")
            if raw_tz in valid_tzs:
                selected_tz = raw_tz
                break
            console.print(f"[red]Unknown timezone: {raw_tz!r}. Try again.[/red]")
    else:
        selected_tz = tz_choices[tz_idx - 1]

    console.print(f"[green]Timezone: {selected_tz}[/green]")

    # ── Build config dict ────────────────────────────────────────
    config: dict[str, Any] = {
        "defaults": {
            "ocr": {"provider": ocr_provider},
            "llm": {"provider": llm_provider, "model": llm_model},
            "embedding": {"provider": embedding_provider},
            "backup": {"provider": backup_provider},
            "timezone": selected_tz,
        },
    }

    if embedding_model is not None:
        config["defaults"]["embedding"]["model"] = embedding_model

    if ollama_base_url is not None:
        config["defaults"]["ollama"] = {
            "base_url": ollama_base_url,
            "model": ollama_model,
        }

    if backup_local_path is not None:
        config["defaults"]["backup"]["local_path"] = backup_local_path

    # ── Write config ─────────────────────────────────────────────
    write_config(config)
    console.print(f"\n[green]Configuration saved to {config_file}[/green]")

    # ── Show env var guidance ────────────────────────────────────
    env_lines: list[str] = []
    env_guidance: list[str] = []

    if ocr_provider == "mistral":
        env_lines.append("MISTRAL_API_KEY=your-key-here")
        env_guidance.append("Mistral OCR requires MISTRAL_API_KEY")

    needs_openai_key = llm_provider == "openai" or embedding_provider == "openai"
    if needs_openai_key:
        env_lines.append("OPENAI_API_KEY=your-key-here")
        uses = []
        if llm_provider == "openai":
            uses.append("LLM")
        if embedding_provider == "openai":
            uses.append("embeddings")
        env_guidance.append(f"OpenAI {' + '.join(uses)} requires OPENAI_API_KEY")

    if llm_provider == "anthropic":
        env_lines.append("ANTHROPIC_API_KEY=your-key-here")
        env_guidance.append("Claude LLM requires ANTHROPIC_API_KEY")

    if env_lines:
        console.print("\n[bold]Required environment variables:[/bold]")
        for g in env_guidance:
            console.print(f"  • {g}")

        env_file = config_dir / ".env"
        console.print(f"\n[dim]Add these to {env_file}:[/dim]")
        console.print("")
        for line in env_lines:
            console.print(f"  {line}")
        console.print("")
    else:
        console.print("\n[green]No API keys required for your setup![/green]")

    console.print("[bold green]Setup complete![/bold green] Run [bold]billfox parse[/bold] to get started.")
