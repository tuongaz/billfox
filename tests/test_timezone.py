"""Tests for timezone resolution helpers."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from billfox.cli._helpers import get_machine_timezone, resolve_timezone_offset


class TestGetMachineTimezone:
    def test_returns_iana_string(self) -> None:
        result = get_machine_timezone()
        # On most dev machines this should return something
        if result is not None:
            assert isinstance(result, str)
            assert "/" in result  # IANA format like "Australia/Sydney"

    @patch("billfox.cli._helpers.datetime")
    def test_returns_none_on_failure(self, mock_dt: object) -> None:
        # Simulate datetime.now() raising
        from unittest.mock import MagicMock

        mock_dt_obj = MagicMock()
        mock_dt_obj.now.side_effect = RuntimeError("no tz")
        # We need to patch the function more carefully
        with patch("billfox.cli._helpers.datetime") as m:
            m.now.side_effect = RuntimeError("no tz")
            result = get_machine_timezone()
        assert result is None


class TestResolveTimezoneOffset:
    def test_none_input_returns_none(self) -> None:
        assert resolve_timezone_offset(None, {}) is None

    def test_already_aware_returns_as_is(self) -> None:
        dt = datetime(2025, 4, 19, 0, 0, 0, tzinfo=timezone(timedelta(hours=10)))
        result = resolve_timezone_offset(dt, {})
        assert result is dt

    def test_naive_with_config_timezone(self) -> None:
        dt = datetime(2025, 4, 19, 0, 0, 0)
        config = {"defaults": {"timezone": "Australia/Sydney"}}
        result = resolve_timezone_offset(dt, config)
        assert result is not None
        assert result.tzinfo == ZoneInfo("Australia/Sydney")
        assert result.year == 2025
        assert result.month == 4
        assert result.day == 19

    def test_naive_with_invalid_config_timezone_falls_to_machine(self) -> None:
        dt = datetime(2025, 4, 19, 0, 0, 0)
        config = {"defaults": {"timezone": "Invalid/Timezone"}}
        with patch("billfox.cli._helpers.get_machine_timezone", return_value="Europe/London"):
            result = resolve_timezone_offset(dt, config)
        assert result is not None
        assert result.tzinfo == ZoneInfo("Europe/London")

    def test_naive_no_config_uses_machine_timezone(self) -> None:
        dt = datetime(2025, 4, 19, 0, 0, 0)
        with patch("billfox.cli._helpers.get_machine_timezone", return_value="America/New_York"):
            result = resolve_timezone_offset(dt, {})
        assert result is not None
        assert result.tzinfo == ZoneInfo("America/New_York")

    def test_naive_no_config_no_machine_returns_naive(self) -> None:
        dt = datetime(2025, 4, 19, 0, 0, 0)
        with patch("billfox.cli._helpers.get_machine_timezone", return_value=None):
            result = resolve_timezone_offset(dt, {})
        assert result is not None
        assert result.tzinfo is None  # Still naive
        assert result == dt
