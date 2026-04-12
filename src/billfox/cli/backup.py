"""CLI command for file backup using configured provider."""

from __future__ import annotations

import asyncio
import mimetypes
import tomllib
from pathlib import Path
from typing import Any

import typer


def build_backup_from_config(
    provider: str | None, local_path: str | None,
) -> Any:
    """Build a backup provider instance from config values.

    Falls back to GoogleDriveBackup when provider is not 'local'.
    """
    if provider == "local":
        from billfox.backup.local import LocalBackup

        if not local_path:
            raise ValueError(
                "defaults.backup.local_path must be set when using local backup provider."
            )
        return LocalBackup(base_path=local_path)

    from billfox.backup.google_drive.client import GoogleDriveBackup

    return GoogleDriveBackup()


def _read_backup_config() -> tuple[str | None, str | None]:
    """Read backup provider and local_path from ~/.billfox/config.toml."""
    config_file = Path.home() / ".billfox" / "config.toml"
    if not config_file.exists():
        return None, None
    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    defaults = config.get("defaults", {})
    backup_cfg = defaults.get("backup", {})
    return backup_cfg.get("provider"), backup_cfg.get("local_path")


def backup(
    files: list[str] = typer.Argument(..., help="File(s) to back up."),  # noqa: B008
) -> None:
    """Back up files using the configured backup provider."""
    from billfox.cli.app import _ensure_configured  # lazy import to avoid circular

    _ensure_configured()

    from rich.console import Console

    console = Console(stderr=True)

    async def _run() -> list[tuple[str, str]]:
        from billfox._types import Document

        provider, local_path = _read_backup_config()
        backup_instance = build_backup_from_config(provider, local_path)
        results: list[tuple[str, str]] = []

        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                console.print(f"[red]Error:[/red] File not found: {file_path}")
                continue

            mime_type, _ = mimetypes.guess_type(path.name)
            document = Document(
                content=path.read_bytes(),
                mime_type=mime_type or "application/octet-stream",
                source_uri=str(path),
            )

            console.print(f"[bold blue]Uploading[/bold blue] {path.name}...", highlight=False)
            result = await backup_instance.backup(document)
            console.print(f"[bold blue]Uploading[/bold blue] {path.name} [green]done[/green]", highlight=False)
            results.append((path.name, result.uri))

        return results

    try:
        results = asyncio.run(_run())
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except ImportError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    for _name, uri in results:
        print(uri)
