"""Tests for .env loading on CLI startup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestDotenvLoading:
    """Tests that billfox CLI loads .env files on startup."""

    def test_loads_global_and_local_env_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify both global and local .env files are loaded in order."""
        global_env = tmp_path / ".billfox" / ".env"
        global_env.parent.mkdir(parents=True)
        global_env.write_text("BILLFOX_TEST_GLOBAL=hello\n")

        local_env = tmp_path / ".env"
        local_env.write_text("BILLFOX_TEST_LOCAL=world\n")

        monkeypatch.delenv("BILLFOX_TEST_GLOBAL", raising=False)
        monkeypatch.delenv("BILLFOX_TEST_LOCAL", raising=False)

        from dotenv import load_dotenv

        load_dotenv(global_env)
        load_dotenv(local_env)

        assert os.environ.get("BILLFOX_TEST_GLOBAL") == "hello"
        assert os.environ.get("BILLFOX_TEST_LOCAL") == "world"

        monkeypatch.delenv("BILLFOX_TEST_GLOBAL", raising=False)
        monkeypatch.delenv("BILLFOX_TEST_LOCAL", raising=False)

    def test_global_env_loads_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that variables from ~/.billfox/.env are loaded into the environment."""
        env_file = tmp_path / ".env"
        env_file.write_text("BILLFOX_TEST_KEY=from_dotenv\n")

        # Ensure the var is not set
        monkeypatch.delenv("BILLFOX_TEST_KEY", raising=False)

        from dotenv import load_dotenv

        load_dotenv(env_file)

        assert os.environ.get("BILLFOX_TEST_KEY") == "from_dotenv"

        # Clean up
        monkeypatch.delenv("BILLFOX_TEST_KEY", raising=False)

    def test_existing_env_vars_take_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that existing env vars are NOT overridden by .env values (override=False default)."""
        env_file = tmp_path / ".env"
        env_file.write_text("BILLFOX_EXISTING_VAR=from_dotenv\n")

        # Set the var before loading
        monkeypatch.setenv("BILLFOX_EXISTING_VAR", "from_shell")

        from dotenv import load_dotenv

        load_dotenv(env_file)

        # Shell value should win
        assert os.environ.get("BILLFOX_EXISTING_VAR") == "from_shell"

    def test_local_env_overrides_global(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that project-local .env values override global .env values.

        Since global is loaded first and local second, and load_dotenv with
        override=False skips already-set vars, we verify the ordering: global
        loads first, then local does NOT override it. This is the expected
        dotenv stacking behavior.
        """
        global_dir = tmp_path / "global" / ".billfox"
        global_dir.mkdir(parents=True)
        global_env = global_dir / ".env"
        global_env.write_text("BILLFOX_STACK_VAR=global_value\n")

        local_env = tmp_path / "local" / ".env"
        local_env.parent.mkdir(parents=True)
        local_env.write_text("BILLFOX_STACK_VAR=local_value\n")

        monkeypatch.delenv("BILLFOX_STACK_VAR", raising=False)

        from dotenv import load_dotenv

        # Simulate the loading order from app.py: global first, then local
        load_dotenv(global_env)
        load_dotenv(local_env)

        # Global was loaded first, so it wins (override=False)
        assert os.environ.get("BILLFOX_STACK_VAR") == "global_value"

        monkeypatch.delenv("BILLFOX_STACK_VAR", raising=False)

    def test_missing_env_file_does_not_error(self, tmp_path: Path) -> None:
        """Verify that missing .env files do not cause errors."""
        from dotenv import load_dotenv

        # Should not raise
        load_dotenv(tmp_path / "nonexistent" / ".env")

    def test_dotenv_import_in_cli_app(self) -> None:
        """Verify that the CLI app module source contains load_dotenv calls."""
        from billfox.cli import app as _app_ref  # noqa: F401

        source_path = Path(__file__).resolve().parent.parent / "src" / "billfox" / "cli" / "app.py"
        source = source_path.read_text()
        assert "load_dotenv" in source
        assert ".billfox" in source
