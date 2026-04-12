# PRD: Initial Setup Wizard, Claude/Ollama LLM Support & Local Backup

## Introduction

Billfox currently requires manual configuration via `billfox config set` commands before use. New users have no guided onboarding, and must discover the correct config keys on their own. Additionally, while the LLM layer (pydantic-ai) technically supports Claude/Anthropic, there is no explicit dependency, documentation, or easy selection. Ollama (local LLM inference) is not supported at all. For backups, Google Drive is the only option — users who want local backups have no alternative.

This PRD introduces four related features:
1. An interactive setup wizard (`billfox init`) that guides first-time configuration
2. Explicit Claude/Anthropic LLM support as an optional dependency
3. Ollama integration for local LLM inference (connect-only mode)
4. A local folder backup provider as an alternative to Google Drive

## Goals

- Provide a zero-to-working experience for new users via `billfox init`
- Detect missing or incomplete configuration and guide users to run `billfox init`
- Support Claude/Anthropic as a first-class LLM option with proper dependency management
- Enable Ollama as a local, free LLM alternative with connectivity validation
- Offer local folder backup as a simple alternative to Google Drive
- Guide users to store API keys in `~/.billfox/.env` rather than in config or shell profiles

## User Stories

### US-001: Run `billfox init` to configure preferences

**Description:** As a new user, I want to run `billfox init` to interactively set up my OCR, LLM, and backup preferences so that I can start using billfox without reading documentation.

**Acceptance Criteria:**
- [ ] `billfox init` command exists as a top-level CLI command
- [ ] Wizard prompts for OCR provider: Docling (local, free) or Mistral (API)
- [ ] Wizard prompts for LLM provider: OpenAI, Claude (Anthropic), or Ollama (local)
- [ ] Wizard prompts for backup provider: Local folder or Google Drive
- [ ] Selections are saved to `~/.billfox/config.toml` using nested TOML tables
- [ ] Running `billfox init` again overwrites previous config (with confirmation)
- [ ] Typecheck/lint passes

### US-002: Receive API key guidance during setup

**Description:** As a user, I want the setup wizard to tell me which API keys I need and where to put them so that I don't have to figure out environment variables on my own.

**Acceptance Criteria:**
- [ ] After selecting providers, wizard displays which env vars are needed (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`)
- [ ] Wizard prints a template `.env` file showing the required variables
- [ ] Wizard tells user to save the file at `~/.billfox/.env`
- [ ] Wizard does NOT ask user to type API keys into the terminal
- [ ] Typecheck/lint passes

### US-003: Load API keys from `.env` file

**Description:** As a user, I want billfox to automatically load API keys from `~/.billfox/.env` so that I can manage my keys in one place.

**Acceptance Criteria:**
- [ ] `python-dotenv` is added as an explicit dependency in the `cli` extra
- [ ] On CLI startup, billfox loads `~/.billfox/.env` (global) and `./.env` (project-local)
- [ ] Env vars from `.env` files are available to all commands (extract, parse, backup, etc.)
- [ ] Existing env vars take precedence over `.env` values (standard dotenv behavior)
- [ ] `.env` is added to `.gitignore`
- [ ] Typecheck/lint passes

### US-004: Detect missing config and suggest `billfox init`

**Description:** As a user who hasn't run setup, I want billfox to tell me to run `billfox init` when I try to use a command that needs configuration, so I don't get cryptic errors.

**Acceptance Criteria:**
- [ ] `extract`, `parse`, and `backup` commands check for config completeness before executing
- [ ] If config is missing or key fields are empty, print a message: "billfox is not configured yet. Run 'billfox init' to set up."
- [ ] Guard only triggers when the user does NOT pass explicit CLI flags (e.g., `--extractor`, `--model`)
- [ ] Exit with code 1 when config is missing
- [ ] Typecheck/lint passes

### US-005: Add Anthropic/Claude as explicit optional dependency

**Description:** As a developer, I want `anthropic` listed as an optional dependency so that `pip install billfox[anthropic]` installs everything needed for Claude.

**Acceptance Criteria:**
- [ ] `pyproject.toml` has new optional extra: `anthropic = ["anthropic>=0.40"]`
- [ ] `anthropic` is included in the `all` extras group
- [ ] `LLMParser` works with model string `anthropic:claude-sonnet-4-20250514` (already works via pydantic-ai, just verify)
- [ ] Typecheck/lint passes

### US-006: Select Claude as default LLM in setup wizard

**Description:** As a user, I want to choose Claude during `billfox init` so that all subsequent `parse` commands use Claude by default.

**Acceptance Criteria:**
- [ ] Selecting "Claude" in the wizard sets `defaults.llm.provider = "anthropic"` and `defaults.llm.model = "anthropic:claude-sonnet-4-20250514"` in config
- [ ] Wizard shows that `ANTHROPIC_API_KEY` is required in the `.env` template
- [ ] `billfox parse` uses the configured model when `--model` is not passed
- [ ] Typecheck/lint passes

### US-007: Support Ollama model prefix in LLMParser

**Description:** As a developer, I want `LLMParser` to handle `ollama:model-name` strings so that Ollama models work through pydantic-ai's OpenAI-compatible provider.

**Acceptance Criteria:**
- [ ] `LLMParser.__init__` accepts an optional `base_url: str | None = None` parameter
- [ ] When model starts with `ollama:`, `_get_agent()` constructs a pydantic-ai `OpenAIModel` with the Ollama base URL (`{base_url}/v1/`)
- [ ] Model names with tags work correctly (e.g., `ollama:llama3.2:7b` splits on first colon only)
- [ ] Default base URL is `http://localhost:11434` when not specified
- [ ] Typecheck/lint passes
- [ ] Tests cover Ollama model resolution (mock pydantic-ai Agent/OpenAIModel)

### US-008: Select Ollama as LLM in setup wizard

**Description:** As a user, I want to choose Ollama during `billfox init` so that I can use my local models without any API keys.

**Acceptance Criteria:**
- [ ] Selecting "Ollama" in wizard prompts for base URL (default: `http://localhost:11434`)
- [ ] Wizard attempts HTTP GET to `{base_url}/api/tags` to verify Ollama is running
- [ ] If connected, wizard lists available models and lets user pick one
- [ ] If not connected, wizard warns but still saves config (user can start Ollama later)
- [ ] Config stores `defaults.llm.provider = "ollama"`, `defaults.ollama.base_url`, and `defaults.ollama.model`
- [ ] No API key guidance shown (Ollama is local, no keys needed)
- [ ] Typecheck/lint passes

### US-009: Wire Ollama config into CLI parse command

**Description:** As a user, I want `billfox parse` to use my configured Ollama settings so that I don't have to pass `--model` and base URL every time.

**Acceptance Criteria:**
- [ ] `parse` command reads `defaults.ollama.base_url` and `defaults.ollama.model` from config
- [ ] When config LLM provider is `ollama`, model string is constructed as `ollama:{model_name}`
- [ ] `base_url` from config is passed to `LLMParser`
- [ ] User can override with `--model ollama:other-model` on the command line
- [ ] Typecheck/lint passes

### US-010: Create LocalBackup provider

**Description:** As a developer, I want a `LocalBackup` class that implements the `DocumentBackup` protocol so that documents can be backed up to a local folder.

**Acceptance Criteria:**
- [ ] `LocalBackup` class in `src/billfox/backup/local.py` implements `DocumentBackup` protocol
- [ ] Takes `base_path: str` in constructor
- [ ] Creates date-based folder structure: `{base_path}/YYYY/MM/DD/`
- [ ] Copies document bytes to the date folder using the original filename
- [ ] Returns `BackupResult` with `uri` pointing to the saved file path and `provider = "local"`
- [ ] If file with same name exists, overwrites it (same behavior as Google Drive deduplication)
- [ ] `LocalBackup` is re-exported from `backup/__init__.py`
- [ ] Typecheck/lint passes
- [ ] Tests cover: folder creation, file writing, date structure, protocol conformance

### US-011: Select local backup in setup wizard

**Description:** As a user, I want to choose local folder backup during `billfox init` so that my receipts are saved to a folder I specify.

**Acceptance Criteria:**
- [ ] Selecting "Local folder" in wizard prompts for a folder path
- [ ] Wizard validates path is writable, offers to create it if it doesn't exist
- [ ] Config stores `defaults.backup.provider = "local"` and `defaults.backup.local_path = "/path/to/folder"`
- [ ] Typecheck/lint passes

### US-012: Wire backup provider selection into CLI

**Description:** As a user, I want `billfox backup` to use my configured backup provider (local or Google Drive) so that it works without extra flags.

**Acceptance Criteria:**
- [ ] `backup` command reads `defaults.backup.provider` from config
- [ ] If `local`, constructs `LocalBackup` with the configured path
- [ ] If `google_drive`, constructs `GoogleDriveBackup` (existing behavior)
- [ ] User can override provider via CLI flag if needed
- [ ] `parse --store` also uses the configured backup provider
- [ ] Typecheck/lint passes

### US-013: Tests for init command

**Description:** As a developer, I want tests for the `billfox init` command so that the wizard flow is verified.

**Acceptance Criteria:**
- [ ] Tests use `CliRunner` with `input=` to simulate user prompts
- [ ] Test: selecting Docling + OpenAI + Local backup writes correct config
- [ ] Test: selecting Mistral + Claude + Google Drive writes correct config
- [ ] Test: selecting Ollama triggers base URL prompt
- [ ] Test: re-running init with existing config asks for confirmation
- [ ] Test: missing config guard triggers on `extract`/`parse`/`backup`
- [ ] Typecheck/lint passes

## Functional Requirements

- FR-1: `billfox init` is a top-level CLI command that launches an interactive setup wizard
- FR-2: The wizard prompts for OCR provider (Docling or Mistral), LLM provider (OpenAI, Claude, or Ollama), and backup provider (Local folder or Google Drive)
- FR-3: All preferences are saved to `~/.billfox/config.toml` using nested TOML tables (`defaults.ocr`, `defaults.llm`, `defaults.ollama`, `defaults.backup`)
- FR-4: The wizard displays required environment variables and prints a `.env` template based on the user's selections
- FR-5: The CLI loads `~/.billfox/.env` and `./.env` on startup using `python-dotenv`
- FR-6: Commands `extract`, `parse`, and `backup` check config completeness and suggest `billfox init` when config is missing (unless explicit CLI flags are passed)
- FR-7: `anthropic` is added as an optional dependency extra in `pyproject.toml`
- FR-8: `LLMParser` supports `ollama:model-name` prefix by constructing a pydantic-ai `OpenAIModel` with Ollama's OpenAI-compatible endpoint (`{base_url}/v1/`)
- FR-9: `LLMParser.__init__` accepts an optional `base_url` parameter for custom API endpoints
- FR-10: The init wizard verifies Ollama connectivity via HTTP GET to `{base_url}/api/tags` and lists available models
- FR-11: `LocalBackup` class implements `DocumentBackup` protocol, saving documents to `{base_path}/YYYY/MM/DD/` folder structure
- FR-12: `billfox backup` reads the configured backup provider from config and constructs the appropriate backup instance
- FR-13: Ollama model names with tags (e.g., `llama3.2:7b`) are handled correctly by splitting on the first colon only

## Non-Goals (Out of Scope)

- No Ollama server lifecycle management (start/stop/install) — billfox only connects to a running instance
- No API key storage in `config.toml` — keys go in `.env` files only
- No web-based setup UI — wizard is terminal-only via Typer prompts
- No automatic provider installation — if `docling` or `anthropic` extras aren't installed, the wizard warns but doesn't run `pip install`
- No model fine-tuning or custom model configuration beyond provider/model name
- No migration tool for existing configs — old flat keys are ignored once new nested keys are set
- No embedding provider selection in the wizard (remains OpenAI-only for now)

## Technical Considerations

- **pydantic-ai Ollama support**: pydantic-ai has no native Ollama provider. Use `OpenAIModel` with `OpenAIProvider(base_url="{ollama_url}/v1/")` since Ollama exposes an OpenAI-compatible API
- **Backward compatibility**: Existing `defaults.extractor` and `defaults.model` flat config keys should be read as fallbacks if the new nested keys are absent
- **Lazy imports**: All new optional dependencies (`anthropic`, `python-dotenv`) must follow the existing lazy import pattern with clear `ImportError` messages
- **Async in CLI**: Ollama connectivity check uses `httpx` (already a transitive dependency) for the HTTP GET. Run via `asyncio.run()` or `asyncio.to_thread()` in the synchronous CLI context
- **Config schema** (new TOML structure):
  ```toml
  [defaults.ocr]
  provider = "docling"  # "docling" | "mistral"

  [defaults.llm]
  provider = "openai"  # "openai" | "anthropic" | "ollama"
  model = "openai:gpt-4.1"  # full pydantic-ai model string

  [defaults.ollama]
  base_url = "http://localhost:11434"
  model = "llama3.2"  # model name without provider prefix

  [defaults.backup]
  provider = "local"  # "local" | "google_drive"
  local_path = "/path/to/backup/folder"
  ```
- **Key files to modify**:
  - `src/billfox/cli/app.py` — dotenv loading, config guard, init command registration, Ollama base_url plumbing in parse
  - `src/billfox/parse/llm.py` — `base_url` parameter, `ollama:` prefix handling
  - `src/billfox/backup/local.py` — new file for `LocalBackup`
  - `src/billfox/backup/__init__.py` — re-export `LocalBackup`
  - `src/billfox/cli/init.py` — new file for the setup wizard
  - `src/billfox/cli/backup.py` — read backup provider from config
  - `pyproject.toml` — new `anthropic` extra, `python-dotenv` in `cli` extra

## Success Metrics

- New users can go from `pip install billfox[all]` to a working `billfox parse` in under 2 minutes
- `billfox init` completes in under 30 seconds for all provider combinations
- Zero API keys stored in `config.toml` — all keys loaded from `.env`
- Ollama users can parse documents without any cloud API keys

## Open Questions

- Should the wizard offer a "test configuration" step at the end that runs a small extraction to verify everything works?
- Should `billfox init` support a `--non-interactive` mode with flags for CI/scripting use cases?
- Should the config guard also check that required packages are installed (e.g., warn if `anthropic` extra is missing when Claude is selected)?
