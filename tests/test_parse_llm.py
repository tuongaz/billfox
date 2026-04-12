from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from billfox.parse._base import Parser
from billfox.parse.llm import LLMParser

# -- Sample Pydantic model for testing --


class Invoice(BaseModel):
    vendor_name: str
    total: float
    date: str | None = None


# -- Helpers --


@dataclass
class FakeRunResult:
    output: Any


# -- Tests --


class TestLLMParserInit:
    def test_stores_params(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract invoice data.",
            retries=3,
        )
        assert parser._model == "openai:gpt-4.1"
        assert parser._output_type is Invoice
        assert parser._system_prompt == "Extract invoice data."
        assert parser._retries == 3

    def test_default_retries(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )
        assert parser._retries == 1


class TestLLMParserImportError:
    def test_raises_clear_import_error(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )
        with patch.dict("sys.modules", {"pydantic_ai": None}), pytest.raises(
            ImportError, match="pydantic-ai is required"
        ):
            parser._get_agent()


class TestLLMParserParse:
    async def test_parse_returns_typed_output(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract invoice data.",
        )

        expected = Invoice(vendor_name="Acme Corp", total=42.50, date="2025-01-15")
        fake_result = FakeRunResult(output=expected)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch.object(parser, "_get_agent", return_value=mock_agent):
            result = await parser.parse("# Invoice\nVendor: Acme Corp\nTotal: $42.50\nDate: 2025-01-15")

        assert result == expected
        assert isinstance(result, Invoice)
        assert result.vendor_name == "Acme Corp"
        assert result.total == 42.50
        assert result.date == "2025-01-15"

    async def test_parse_passes_markdown_to_agent(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract invoice data.",
        )

        fake_result = FakeRunResult(output=Invoice(vendor_name="X", total=0))
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        markdown = "Some markdown content"
        with patch.object(parser, "_get_agent", return_value=mock_agent):
            await parser.parse(markdown)

        mock_agent.run.assert_called_once_with(markdown)

    async def test_parse_with_none_optional_fields(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )

        expected = Invoice(vendor_name="Shop", total=10.0, date=None)
        fake_result = FakeRunResult(output=expected)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch.object(parser, "_get_agent", return_value=mock_agent):
            result = await parser.parse("Vendor: Shop\nTotal: $10")

        assert result.date is None


class TestLLMParserGetAgent:
    def test_get_agent_passes_params(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract invoice.",
            retries=2,
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            parser._get_agent()
            MockAgent.assert_called_once_with(
                "openai:gpt-4.1",
                output_type=Invoice,
                system_prompt="Extract invoice.",
                retries=2,
            )


class TestLLMParserAnthropic:
    def test_anthropic_model_string_passed_to_agent(self) -> None:
        parser = LLMParser(
            model="anthropic:claude-sonnet-4-20250514",
            output_type=Invoice,
            system_prompt="Extract invoice data.",
        )
        assert parser._model == "anthropic:claude-sonnet-4-20250514"

        with patch("pydantic_ai.Agent") as MockAgent:
            parser._get_agent()
            MockAgent.assert_called_once_with(
                "anthropic:claude-sonnet-4-20250514",
                output_type=Invoice,
                system_prompt="Extract invoice data.",
                retries=1,
            )

    async def test_anthropic_model_parse_returns_output(self) -> None:
        parser = LLMParser(
            model="anthropic:claude-sonnet-4-20250514",
            output_type=Invoice,
            system_prompt="Extract invoice data.",
        )

        expected = Invoice(vendor_name="Acme Corp", total=99.99, date="2025-06-01")
        fake_result = FakeRunResult(output=expected)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch.object(parser, "_get_agent", return_value=mock_agent):
            result = await parser.parse("# Invoice\nVendor: Acme Corp\nTotal: $99.99")

        assert result == expected
        assert isinstance(result, Invoice)


class TestLLMParserOllama:
    def test_ollama_prefix_creates_openai_model(self) -> None:
        parser = LLMParser(
            model="ollama:llama3.2",
            output_type=Invoice,
            system_prompt="Extract.",
        )

        with (
            patch("pydantic_ai.Agent") as MockAgent,
            patch(
                "pydantic_ai.models.openai.OpenAIModel"
            ) as MockOpenAIModel,
            patch(
                "pydantic_ai.providers.openai.OpenAIProvider"
            ) as MockOpenAIProvider,
        ):
            parser._get_agent()
            MockOpenAIProvider.assert_called_once_with(
                base_url="http://localhost:11434/v1/"
            )
            MockOpenAIModel.assert_called_once_with(
                "llama3.2",
                provider=MockOpenAIProvider.return_value,
            )
            MockAgent.assert_called_once_with(
                MockOpenAIModel.return_value,
                output_type=Invoice,
                system_prompt="Extract.",
                retries=1,
            )

    def test_ollama_model_with_tag(self) -> None:
        parser = LLMParser(
            model="ollama:llama3.2:7b",
            output_type=Invoice,
            system_prompt="Extract.",
        )

        with (
            patch("pydantic_ai.Agent"),
            patch(
                "pydantic_ai.models.openai.OpenAIModel"
            ) as MockOpenAIModel,
            patch("pydantic_ai.providers.openai.OpenAIProvider"),
        ):
            parser._get_agent()
            # Split on first colon only: model name should be "llama3.2:7b"
            assert MockOpenAIModel.call_args[0][0] == "llama3.2:7b"

    def test_ollama_custom_base_url(self) -> None:
        parser = LLMParser(
            model="ollama:llama3.2",
            output_type=Invoice,
            system_prompt="Extract.",
            base_url="http://myserver:11434",
        )

        with (
            patch("pydantic_ai.Agent"),
            patch("pydantic_ai.models.openai.OpenAIModel"),
            patch(
                "pydantic_ai.providers.openai.OpenAIProvider"
            ) as MockOpenAIProvider,
        ):
            parser._get_agent()
            MockOpenAIProvider.assert_called_once_with(
                base_url="http://myserver:11434/v1/"
            )

    def test_ollama_default_base_url_when_none(self) -> None:
        parser = LLMParser(
            model="ollama:llama3.2",
            output_type=Invoice,
            system_prompt="Extract.",
            base_url=None,
        )

        with (
            patch("pydantic_ai.Agent"),
            patch("pydantic_ai.models.openai.OpenAIModel"),
            patch(
                "pydantic_ai.providers.openai.OpenAIProvider"
            ) as MockOpenAIProvider,
        ):
            parser._get_agent()
            MockOpenAIProvider.assert_called_once_with(
                base_url="http://localhost:11434/v1/"
            )

    def test_non_ollama_model_passthrough(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )

        with patch("pydantic_ai.Agent") as MockAgent:
            parser._get_agent()
            MockAgent.assert_called_once_with(
                "openai:gpt-4.1",
                output_type=Invoice,
                system_prompt="Extract.",
                retries=1,
            )

    def test_base_url_stored(self) -> None:
        parser = LLMParser(
            model="ollama:llama3.2",
            output_type=Invoice,
            system_prompt="Extract.",
            base_url="http://custom:1234",
        )
        assert parser._base_url == "http://custom:1234"

    def test_base_url_defaults_to_none(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )
        assert parser._base_url is None


class TestLLMParserProtocol:
    def test_conforms_to_parser_protocol(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )
        assert isinstance(parser, Parser)

    def test_has_parse_method(self) -> None:
        parser = LLMParser(
            model="openai:gpt-4.1",
            output_type=Invoice,
            system_prompt="Extract.",
        )
        assert hasattr(parser, "parse")
        assert callable(parser.parse)
