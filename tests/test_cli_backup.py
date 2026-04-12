"""Tests for CLI backup provider selection (US-005)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from billfox.cli.app import app
from billfox.cli.backup import build_backup_from_config

runner = CliRunner()


# ---------------------------------------------------------------------------
# build_backup_from_config
# ---------------------------------------------------------------------------


class TestBuildBackupFromConfig:
    """Tests for the backup provider factory function."""

    def test_local_provider_returns_local_backup(self, tmp_path: Path) -> None:
        result = build_backup_from_config("local", str(tmp_path))
        from billfox.backup.local import LocalBackup

        assert isinstance(result, LocalBackup)

    def test_local_provider_without_path_raises(self) -> None:
        with pytest.raises(ValueError, match="defaults.backup.local_path"):
            build_backup_from_config("local", None)

    def test_local_provider_with_empty_path_raises(self) -> None:
        with pytest.raises(ValueError, match="defaults.backup.local_path"):
            build_backup_from_config("local", "")

    @patch("billfox.backup.google_drive.client.GoogleDriveBackup")
    def test_google_drive_provider(self, mock_gdrive_cls: MagicMock) -> None:
        result = build_backup_from_config("google_drive", None)
        mock_gdrive_cls.assert_called_once()
        assert result is mock_gdrive_cls.return_value

    @patch("billfox.backup.google_drive.client.GoogleDriveBackup")
    def test_none_provider_falls_back_to_google_drive(
        self, mock_gdrive_cls: MagicMock,
    ) -> None:
        result = build_backup_from_config(None, None)
        mock_gdrive_cls.assert_called_once()
        assert result is mock_gdrive_cls.return_value


# ---------------------------------------------------------------------------
# backup command provider selection
# ---------------------------------------------------------------------------


class TestBackupCommandProviderSelection:
    """Tests that the backup command uses the configured provider."""

    def test_backup_uses_local_when_configured(self, tmp_path: Path) -> None:
        test_file = tmp_path / "invoice.pdf"
        test_file.write_bytes(b"fake pdf")

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[defaults.backup]\n"
            f'provider = "local"\n'
            f'local_path = "{backup_dir}"\n'
        )

        with (
            patch("billfox.cli.app._ensure_configured"),
            patch(
                "billfox.cli.backup._read_backup_config",
                return_value=("local", str(backup_dir)),
            ),
        ):
            result = runner.invoke(app, ["backup", str(test_file)])

        assert result.exit_code == 0
        # Verify a file was created in the backup dir (date subfolder)
        backed_up = list(backup_dir.rglob("invoice.pdf"))
        assert len(backed_up) == 1

    @patch("billfox.backup.google_drive.client.GoogleDriveBackup")
    def test_backup_uses_google_drive_when_no_config(
        self, mock_gdrive_cls: MagicMock, tmp_path: Path,
    ) -> None:
        test_file = tmp_path / "invoice.pdf"
        test_file.write_bytes(b"fake pdf")

        mock_instance = mock_gdrive_cls.return_value
        mock_instance.backup = AsyncMock(
            return_value=MagicMock(uri="gdrive://file123"),
        )

        with (
            patch("billfox.cli.app._ensure_configured"),
            patch(
                "billfox.cli.backup._read_backup_config",
                return_value=(None, None),
            ),
        ):
            result = runner.invoke(app, ["backup", str(test_file)])

        assert result.exit_code == 0
        mock_instance.backup.assert_awaited_once()

    @patch("billfox.backup.google_drive.client.GoogleDriveBackup")
    def test_backup_falls_back_to_google_drive_for_unknown_provider(
        self, mock_gdrive_cls: MagicMock, tmp_path: Path,
    ) -> None:
        test_file = tmp_path / "invoice.pdf"
        test_file.write_bytes(b"fake pdf")

        mock_instance = mock_gdrive_cls.return_value
        mock_instance.backup = AsyncMock(
            return_value=MagicMock(uri="gdrive://file123"),
        )

        with (
            patch("billfox.cli.app._ensure_configured"),
            patch(
                "billfox.cli.backup._read_backup_config",
                return_value=("google_drive", None),
            ),
        ):
            result = runner.invoke(app, ["backup", str(test_file)])

        assert result.exit_code == 0
        mock_instance.backup.assert_awaited_once()


# ---------------------------------------------------------------------------
# parse command backup integration
# ---------------------------------------------------------------------------


class TestParseCommandBackupIntegration:
    """Tests that parse --store also wires up backup from config."""

    def test_parse_with_store_passes_backup_to_pipeline(
        self, tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        db_path = tmp_path / "test.db"

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline) as mock_cls,
            patch(
                "billfox.cli.app._read_config",
                return_value={
                    "defaults": {
                        "ocr": {"provider": "docling"},
                        "backup": {
                            "provider": "local",
                            "local_path": str(tmp_path / "backups"),
                        },
                    },
                },
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--store", str(db_path),
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("backup") is not None

    def test_parse_without_store_no_backup(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline) as mock_cls,
            patch(
                "billfox.cli.app._read_config",
                return_value={"defaults": {"ocr": {"provider": "docling"}}},
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("backup") is None

    @patch("billfox.backup.google_drive.client.GoogleDriveBackup")
    def test_parse_with_store_uses_google_drive_fallback(
        self, mock_gdrive_cls: MagicMock, tmp_path: Path,
    ) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        db_path = tmp_path / "test.db"

        schema_file = tmp_path / "schema.py"
        schema_file.write_text(
            "from pydantic import BaseModel\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"name": "Widget"}

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("billfox.pipeline.Pipeline", return_value=mock_pipeline) as mock_cls,
            patch(
                "billfox.cli.app._read_config",
                return_value={"defaults": {"ocr": {"provider": "docling"}}},
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "parse", str(img),
                    "--schema", f"{schema_file}:Item",
                    "--store", str(db_path),
                ],
            )

        assert result.exit_code == 0
        call_kwargs = mock_cls.call_args
        backup_arg = call_kwargs.kwargs.get("backup")
        assert backup_arg is not None
        assert backup_arg is mock_gdrive_cls.return_value
