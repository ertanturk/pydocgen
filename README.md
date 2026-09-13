# pydocgen

AI-assisted Google-style docstring generator for Python source files.

`pydocgen` analyzes Python source code via AST, partitions undocumented functions into token-budgeted batches, synthesizes Google-style docstrings using the Google Gemini API, validates output against AST source truth, and applies edits atomically.

---

<p align="center">
  <img src="demo.gif" alt="PyDocGen Demo" width="850">
</p>

---

## Features

- **AST-driven Extraction**: Accurately extracts top-level functions, methods, classmethods, staticmethods, and nested closures while preserving existing docstrings.
- **Token-aware Batching**: Groups functions dynamically into token-bounded requests to maximize throughput while preventing prompt overflow.
- **Semantic AST Validation**: Cross-checks generated docstrings against the AST to reject hallucinated arguments, verify return semantics, strip `self`/`cls`, and validate explicit exception paths.
- **Atomic Rewriting**: Calculates insertions in reverse line order to prevent offset drift and replaces files via atomic swaps with unified diff previews.
- **Secure Credential Storage**: Integrates with system keyrings (macOS Keychain, Linux Secret Service, Windows Credential Vault) alongside environment variable fallbacks.
- **POSIX Exit Codes**: Adheres to standard `sysexits.h` conventions for scripting and CI pipeline reliability.

---

## Requirements

- Python >= 3.14
- Google Gemini API key

---

## Installation

Install using `uv`:

```bash
# Global CLI tool
uv tool install git+https://github.com/ertanturk/pydocgen.git

# Local development environment
git clone https://github.com/ertanturk/pydocgen.git
cd pydocgen
uv sync
```

Alternatively, install using `pip`:

```bash
pip install .
```

---

## Authentication

Configure your Gemini API key in the OS keyring:

```bash
# Store securely via masked prompt
pydocgen auth set

# Inspect active authentication source
pydocgen auth status

# Remove stored credential
pydocgen auth delete
```

Alternatively, set the environment variable:

```bash
export GEMINI_API_KEY="AIzaSy..."
```

---

## Usage

### Interactive Mode

Inspect undocumented functions, review a colorized unified diff, and choose whether to apply changes:

```bash
pydocgen path/to/module.py
```

### Automatic Apply

Bypass interactive prompts and write validated docstrings directly to disk:

```bash
pydocgen path/to/module.py --apply
```

### Dry Run

Preview unified diffs and validation summaries without modifying files:

```bash
pydocgen path/to/module.py --dry-run
```

### CI Audit Mode

Check whether any functions lack docstrings without invoking the LLM or modifying files:

```bash
pydocgen path/to/module.py --check
```

Exits with code `0` if all functions are documented, or `1` if undocumented functions exist.

---

## Command-Line Interface

```text
pydocgen [COMMAND] PATH [OPTIONS]
```

### Options

| Option                   | Type       | Default            | Description                                               |
| ------------------------ | ---------- | ------------------ | --------------------------------------------------------- |
| `PATH`                   | Positional | Required           | Target Python source file (`.py`).                        |
| `-a`, `--apply`, `-y`    | Flag       | `False`            | Apply changes without confirmation.                       |
| `--check`                | Flag       | `False`            | Audit mode; exits with 1 if undocumented functions exist. |
| `--dry-run`              | Flag       | `False`            | Display proposed diff without writing to disk.            |
| `--overwrite`, `--force` | Flag       | `False`            | Regenerate docstrings for already-documented functions.   |
| `-m`, `--model`          | String     | `gemini-3.8-flash` | Gemini model identifier.                                  |
| `--api-key`              | String     | None               | Runtime API key override.                                 |
| `-c`, `--concurrency`    | Integer    | `3`                | Maximum concurrent batch requests (1-10).                 |
| `--max-batch-tokens`     | Integer    | `12000`            | Token limit ceiling per request batch.                    |
| `--no-strict-raises`     | Flag       | `False`            | Bypass strict AST matching for raised exceptions.         |
| `-q`, `--quiet`          | Flag       | `False`            | Suppress non-essential console output and banners.        |
| `-v`, `--version`        | Flag       | `False`            | Display version information and exit.                     |

---

## Exit Codes

`pydocgen` returns standard BSD/POSIX exit codes:

| Code | Name             | Meaning                                                     |
| ---- | ---------------- | ----------------------------------------------------------- |
| `0`  | `EX_OK`          | Successful execution or check passed.                       |
| `1`  | `CHECK_FAILED`   | Undocumented functions found during `--check`.              |
| `64` | `EX_USAGE`       | Invalid CLI syntax, empty key, or incorrect arguments.      |
| `65` | `EX_DATAERR`     | Python syntax error in input file or invalid model schema.  |
| `66` | `EX_NOINPUT`     | Input file does not exist, is unreadable, or is not `.py`.  |
| `69` | `EX_UNAVAILABLE` | Remote Gemini API unreachable or network connection failed. |
| `70` | `EX_SOFTWARE`    | Internal unexpected error.                                  |
| `75` | `EX_TEMPFAIL`    | API rate limit reached or request timed out.                |
| `78` | `EX_CONFIG`      | API key missing, unconfigured, or invalid.                  |

---

## Development

Run tests and linting using `uv`:

```bash
# Execute test suite
uv run pytest

# Run linter and formatter checks
uv run ruff check .
uv run ruff format --check .

# Run static type checker
uv run ty check
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
