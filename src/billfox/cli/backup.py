"""CLI command for manual file backup to Google Drive."""

from __future__ import annotations

import asyncio
import mimetypes
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


typer = _lazy_typer()


def backup(
    files: list[str] = typer.Argument(..., help="File(s) to back up to Google Drive."),  # noqa: B008
) -> None:
    """Back up files to Google Drive."""
    from rich.console import Console

    console = Console(stderr=True)

    async def _run() -> list[tuple[str, str]]:
        from billfox._types import Document
        from billfox.backup.google_drive.client import GoogleDriveBackup

        drive = GoogleDriveBackup()
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
            result = await drive.backup(document)
            console.print(f"[bold blue]Uploading[/bold blue] {path.name} [green]done[/green]", highlight=False)
            results.append((path.name, result.uri))

        return results

    try:
        results = asyncio.run(_run())
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except ImportError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    for _name, uri in results:
        print(uri)
