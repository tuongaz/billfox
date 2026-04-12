"""CLI commands for authentication with external services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

auth_app: Any = typer.Typer(
    name="auth",
    help="Manage authentication for external services.",
    no_args_is_help=True,
)


@auth_app.command("google-drive")  # type: ignore[untyped-decorator]
def google_drive(
    force: bool = typer.Option(False, "--force", "-f", help="Re-authorize even if already authorized."),
) -> None:
    """Authorize billfox to access your Google Drive."""
    from rich.console import Console

    console = Console(stderr=True)

    from billfox.backup.google_drive.auth import CREDENTIALS_FILE

    if CREDENTIALS_FILE.exists() and not force and not typer.confirm(
        "Google Drive is already authorized. Re-authorize?"
    ):
        raise typer.Exit()

    try:
        from billfox.backup.google_drive.auth import GoogleDriveAuth

        auth = GoogleDriveAuth()
        console.print("[bold blue]Opening browser for Google Drive authorization...[/bold blue]")
        credentials = auth.authorize()
    except ImportError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    # Try to get user email from the credentials
    email = _get_google_email(credentials)
    if email:
        console.print(f"[green]Successfully authorized Google Drive for {email}[/green]")
    else:
        console.print("[green]Successfully authorized Google Drive[/green]")


def _get_google_email(credentials: Any) -> str | None:
    """Try to get the Google account email from credentials."""
    try:
        from googleapiclient.discovery import build

        service = build("oauth2", "v2", credentials=credentials)
        user_info = service.userinfo().get().execute()
        return user_info.get("email")  # type: ignore[no-any-return]
    except Exception:
        return None


@auth_app.command("status")  # type: ignore[untyped-decorator]
def status() -> None:
    """Show which integrations are authorized."""
    from rich.console import Console

    console = Console(stderr=True)

    integrations: list[tuple[str, Path]] = [
        ("Google Drive", Path.home() / ".billfox" / "credentials" / "google_drive.json"),
    ]

    console.print("[bold]Authorization status:[/bold]")
    for name, cred_path in integrations:
        if cred_path.exists():
            console.print(f"  [green]✓[/green] {name}")
        else:
            console.print(f"  [red]✗[/red] {name} — run [bold]billfox auth google-drive[/bold] to authorize")
