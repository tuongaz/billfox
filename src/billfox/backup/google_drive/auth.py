"""Google Drive OAuth flow and credentials management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS_DIR = Path.home() / ".billfox" / "credentials"
CREDENTIALS_FILE = CREDENTIALS_DIR / "google_drive.json"


def _import_google_auth() -> Any:
    """Lazily import google.oauth2.credentials."""
    try:
        from google.oauth2.credentials import Credentials
    except ImportError:
        raise ImportError(
            "google-auth is required for Google Drive backup. "
            "Install it with: pip install 'billfox[google-drive]'"
        ) from None
    return Credentials


def _import_installed_app_flow() -> Any:
    """Lazily import InstalledAppFlow."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise ImportError(
            "google-auth-oauthlib is required for Google Drive backup. "
            "Install it with: pip install 'billfox[google-drive]'"
        ) from None
    return InstalledAppFlow


class GoogleDriveAuth:
    """Handles Google Drive OAuth flow for CLI-based authorization."""

    def __init__(self) -> None:
        self.client_id = os.getenv("GOOGLE_DRIVE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("GOOGLE_DRIVE_CLIENT_ID and GOOGLE_DRIVE_CLIENT_SECRET must be set")

    def authorize(self) -> Any:
        """Run the OAuth flow via a local HTTP server and save credentials.

        Opens a browser for the user to authorize, receives the callback
        on a temporary local server, and saves credentials to disk.

        Returns:
            google.oauth2.credentials.Credentials object.
        """
        InstalledAppFlow = _import_installed_app_flow()

        client_config = {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }

        flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
        credentials = flow.run_local_server(port=0, access_type="offline", prompt="consent")

        self._save_credentials(credentials)
        return credentials

    def _save_credentials(self, credentials: Any) -> None:
        """Save credentials to ~/.billfox/credentials/google_drive.json with 0600 permissions."""
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

        cred_data = {
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "expires_at": credentials.expiry.isoformat() if credentials.expiry else None,
        }

        CREDENTIALS_FILE.write_text(json.dumps(cred_data, indent=2))
        CREDENTIALS_FILE.chmod(0o600)


def load_credentials(credentials_path: str | None = None) -> Any:
    """Load saved Google Drive credentials, refreshing if expired.

    Args:
        credentials_path: Optional path to credentials file.
            Defaults to ~/.billfox/credentials/google_drive.json.

    Returns:
        google.oauth2.credentials.Credentials object.

    Raises:
        FileNotFoundError: If credentials file is missing.
    """
    Credentials = _import_google_auth()

    path = Path(credentials_path) if credentials_path else CREDENTIALS_FILE
    if not path.exists():
        raise FileNotFoundError("Google Drive not authorized. Run 'billfox auth google-drive' to authorize.")

    cred_data = json.loads(path.read_text())

    credentials = Credentials(
        token=cred_data.get("access_token"),
        refresh_token=cred_data.get("refresh_token"),
        token_uri=cred_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=cred_data.get("client_id"),
        client_secret=cred_data.get("client_secret"),
    )

    if credentials.expired and credentials.refresh_token:
        try:
            from google.auth.transport.requests import Request
        except ImportError:
            raise ImportError(
                "google-auth is required for Google Drive backup. "
                "Install it with: pip install 'billfox[google-drive]'"
            ) from None
        credentials.refresh(Request())

    return credentials
