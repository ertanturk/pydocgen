from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from pydocgen import __version__
from pydocgen.application.service import DocumentationService
from pydocgen.cli import output
from pydocgen.config.settings import (
    DEFAULT_MAX_BATCH_TOKENS,
    DEFAULT_MAX_CONCURRENCY,
)
from pydocgen.errors.exceptions import (
    CredentialAlreadyExistsError,
    CredentialNotFoundError,
    ExitCode,
    InvalidCredentialError,
    PydocgenError,
)
from pydocgen.providers.gemini import DEFAULT_MODEL
from pydocgen.security.credentials import Credentials
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="pydocgen",
    help="AI-assisted, deterministic Python docstring generation CLI.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

auth_app = typer.Typer(
    name="auth",
    help="Manage Gemini API credentials stored in your OS keyring.",
    no_args_is_help=True,
)
app.add_typer(auth_app, name="auth")


def version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        output.err_console.print(f"pydocgen version: {__version__}")
        raise typer.Exit(code=ExitCode.OK)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """pydocgen root command group."""
    pass


# --- Auth Subcommands ---


@auth_app.command("set")
def auth_set(
    key: Annotated[
        str | None,
        typer.Option(
            "--key", "-k", help="API key string. If omitted, prompts securely without echoing."
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            "-f",
            help="Overwrite existing key if present in the secure OS keyring.",
        ),
    ] = True,
) -> None:
    """Persist the Gemini API key to the secure OS keyring."""
    api_key = key
    if not api_key:
        api_key = typer.prompt("Enter Gemini API Key", hide_input=True).strip()

    if not api_key:
        output.render_error("Validation", "API key cannot be empty.")
        raise typer.Exit(code=ExitCode.USAGE)

    try:
        logger.info("Persisting Gemini API key to OS keyring", extra={"force": force})
        Credentials.save_api_key(api_key, overwrite=force)
        logger.info("Successfully persisted Gemini API key to OS keyring")
        output.err_console.print(
            "[status.accepted]✓ Successfully stored Gemini API key in OS keyring.[/]"
        )
    except typer.Exit, typer.Abort:
        raise
    except InvalidCredentialError as exc:
        logger.warning("Invalid API key provided: %s", exc)
        output.render_error("Validation", str(exc))
        raise typer.Exit(code=ExitCode.USAGE) from exc
    except CredentialAlreadyExistsError as exc:
        logger.warning("API key already exists in keyring: %s", exc)
        output.render_error("Conflict", f"{exc} Pass --force to overwrite.")
        raise typer.Exit(code=ExitCode.CONFIG) from exc
    except Exception as exc:
        logger.error("Failed to save API key to OS keyring: %s", exc, exc_info=True)
        output.render_error("Keyring Error", str(exc))
        raise typer.Exit(code=ExitCode.SOFTWARE) from exc


@auth_app.command("status")
def auth_status() -> None:
    """Display the active authentication source without exposing the key."""
    logger.debug("Checking active authentication source")
    # 1. Keyring check (primary credential source for Credentials.get_api_key)
    try:
        keyring_key = Credentials.get_api_key(include_env=False)
    except Exception:
        keyring_key = None

    if keyring_key:
        masked = Credentials.mask_api_key(keyring_key)
        logger.debug("Authentication source found in OS keyring")
        output.err_console.print(
            f"Source: [brand]OS Keyring[/] (Account: {Credentials.account_name}, Key: {masked})"
        )
        return

    # 2. Environment variables fallback
    for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            masked = Credentials.mask_api_key(env_val)
            logger.debug("Authentication source found in environment variable: %s", env_var)
            output.err_console.print(f"Source: [brand]Environment Variable[/] ({env_var}={masked})")
            return

    logger.debug("No authentication source found")
    output.err_console.print(
        "[status.skipped]No API key found.[/] Configure one using 'pydocgen auth set' or set GEMINI_API_KEY."
    )


@auth_app.command("delete")
def auth_delete() -> None:
    """Remove stored Gemini API key from the OS keyring."""
    try:
        logger.info("Removing Gemini API key from OS keyring")
        Credentials.delete_api_key()
        logger.info("Successfully removed Gemini API key from OS keyring")
        output.err_console.print("[status.accepted]✓ Gemini API key removed from OS keyring.[/]")
    except typer.Exit, typer.Abort:
        raise
    except CredentialNotFoundError:
        logger.debug("No stored Gemini API key found to delete")
        output.err_console.print("[status.skipped]No stored Gemini API key found in OS keyring.[/]")
    except Exception as exc:
        logger.error("Failed to remove Gemini API key from OS keyring: %s", exc, exc_info=True)
        output.render_error("Keyring Error", str(exc))
        raise typer.Exit(code=ExitCode.SOFTWARE) from exc


# --- Primary Command: pydocgen [PATH] ---


@app.command(name="document")
def document(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to the target Python source file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=False,
        ),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply", "-a", "-y", help="Automatically apply docstrings without interactive prompt."
        ),
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Non-modifying audit mode. Exits with 1 if undocumented functions exist.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Preview proposed docstring changes without modifying files on disk."
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            "--force",
            help="Regenerate docstrings even for already-documented functions.",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Gemini model identifier."),
    ] = DEFAULT_MODEL,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key", help="One-time API key override. Bypasses keyring and environment."
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency", "-c", min=1, max=10, help="Maximum concurrent asynchronous batches."
        ),
    ] = DEFAULT_MAX_CONCURRENCY,
    max_batch_tokens: Annotated[
        int,
        typer.Option("--max-batch-tokens", min=1000, help="Maximum input token ceiling per batch."),
    ] = DEFAULT_MAX_BATCH_TOKENS,
    no_strict_raises: Annotated[
        bool,
        typer.Option(
            "--no-strict-raises", help="Disable strict AST validation for documented exceptions."
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress visual chrome, spinners, and metrics."),
    ] = False,
) -> None:
    logger.info(
        "Invoking document command",
        extra={
            "path": str(path),
            "model": model,
            "concurrency": concurrency,
            "check": check,
            "apply": apply,
            "dry_run": dry_run,
            "overwrite": overwrite,
        },
    )

    if path.suffix != ".py":
        logger.warning("Invalid input file: '%s' is not a Python source file", path)
        output.render_error("Invalid Input", f"'{path.name}' is not a Python (.py) source file.")
        raise typer.Exit(code=ExitCode.NOINPUT)

    # Fast-path for --check mode
    if check:
        _handle_check_mode(path, quiet=quiet)
        return

    if not quiet:
        output.render_banner(path, model)

    service = DocumentationService(
        api_key=api_key,
        model=model,
        max_concurrency=concurrency,
        max_batch_tokens=max_batch_tokens,
        strict_raises=not no_strict_raises,
    )

    try:
        _run_generation_pipeline(
            service,
            path,
            apply=apply,
            dry_run=dry_run,
            overwrite=overwrite,
            quiet=quiet,
        )
    except typer.Exit, typer.Abort:
        raise
    except PydocgenError as exc:
        logger.error("Pipeline failure: %s", exc)
        output.render_error("Pipeline Failure", str(exc))
        raise typer.Exit(code=exc.exit_code) from exc
    except Exception as exc:
        logger.error("Unexpected CLI internal error: %s", exc, exc_info=True)
        output.render_error("Internal Error", str(exc))
        raise typer.Exit(code=ExitCode.SOFTWARE) from exc


def _handle_check_mode(path: Path, *, quiet: bool = False) -> None:
    """Evaluate docstring coverage and return POSIX exit codes without modification."""
    from pydocgen.analysis.extractor import extract_functions
    from pydocgen.analysis.parser import parse_file

    logger.debug("Checking docstring coverage for %s", path)
    try:
        _, source, tree = parse_file(path, enforce_relative=False)
    except typer.Exit, typer.Abort:
        raise
    except PydocgenError as exc:
        logger.warning("Check mode parse failure: %s", exc)
        if not quiet:
            output.render_error("Parse Error", str(exc))
        raise typer.Exit(code=exc.exit_code) from exc

    with logger.timed("cli.check_mode"):
        funcs = extract_functions(source, tree)
        undocumented = [f for f in funcs if not f.has_docstring]

    logger.info(
        "Coverage check completed",
        extra={
            "total_functions": len(funcs),
            "undocumented_functions": len(undocumented),
        },
    )

    if not funcs:
        if not quiet:
            output.err_console.print(f"[status.accepted]No functions found in '{path.name}'.[/]")
        raise typer.Exit(code=ExitCode.OK)

    if not undocumented:
        if not quiet:
            output.err_console.print(
                f"[status.accepted]  All {len(funcs)} function(s) in '{path.name}'"
                " have docstrings.[/]"
            )
        raise typer.Exit(code=ExitCode.OK)

    if not quiet:
        output.err_console.print(
            f"[status.skipped]{len(undocumented)}/{len(funcs)} function(s) lack"
            f" docstrings in '{path.name}'.[/]\nRun 'pydocgen {path.name} --apply'"
            " to generate and apply them."
        )
    raise typer.Exit(code=1)


def _run_generation_pipeline(
    service: DocumentationService,
    path: Path,
    *,
    apply: bool,
    dry_run: bool = False,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Execute the pipeline with live CLI feedback."""
    import asyncio
    import io

    stream = io.StringIO() if quiet else output.err_console.file

    with logger.timed("cli.run_generation_pipeline"):
        logger.info("Executing documentation pipeline for %s", path)
        report = asyncio.run(
            service.document_file(
                path=path,
                interactive=not apply and not dry_run,
                dry_run=dry_run,
                overwrite_existing=overwrite,
                stream=stream,  # type: ignore
                on_analysis_complete=output.render_analysis_stats if not quiet else None,
                on_batch_start=(
                    (
                        lambda b_id, count, cur, tot: output.render_batch_step(
                            b_id, count, cur, tot, "generating"
                        )
                    )
                    if not quiet
                    else None
                ),
                on_batch_complete=(
                    (
                        lambda b_id, count, cur, tot, status: output.render_batch_step(
                            b_id, count, cur, tot, status
                        )
                    )
                    if not quiet
                    else None
                ),
                on_validation_complete=output.render_results_summary if not quiet else None,
                on_preview_diff=output.render_unified_diff if not quiet else None,
                on_prompt_confirmation=output.prompt_confirmation if not quiet else None,
            )
        )

    logger.info(
        "Pipeline run finished",
        extra={
            "path": str(path),
            "total_functions": report.total_functions,
            "to_document": report.to_document,
            "accepted": report.accepted_count,
            "skipped": report.skipped_count,
            "rejected": report.rejected_count,
            "failed_batches": report.failed_batches,
            "applied": report.applied,
        },
    )

    if report.failed_batches > 0:
        logger.error(
            "Batch generation errors encountered: %d failed batches",
            report.failed_batches,
            extra={"batch_errors": report.batch_errors},
        )
        if not quiet and report.batch_errors:
            output.render_error("Batch Generation Errors", report.batch_errors)
        raise typer.Exit(code=ExitCode.TEMPFAIL)

    if report.applied:
        output.err_console.print(f"[status.accepted]✓ Successfully updated '{path.name}'.[/]")
    elif dry_run:
        output.err_console.print(
            f"[status.skipped]Dry run complete: '{path.name}' was not modified.[/]"
        )
    elif report.to_document == 0:
        if not quiet:
            output.err_console.print(
                f"[status.accepted]No functions in '{path.name}' required documentation.[/]"
            )
    elif report.accepted_count == 0:
        output.err_console.print(
            f"[status.skipped]No changes applied to '{path.name}': generated docstrings did not pass semantic validation.[/]"
        )
    elif not apply:
        output.err_console.print("[status.skipped]Changes discarded. No files were modified.[/]")


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint with smart command routing.

    Routes bare file paths and option flags directly to the 'document' command
    unless a known root command ('auth', 'document') or help/version flag is given.
    """
    argv = list(sys.argv[1:]) if args is None else list(args)

    exempt_commands = {
        "auth",
        "document",
        "--help",
        "-h",
        "--version",
        "-v",
        "--install-completion",
        "--show-completion",
    }

    if argv and argv[0] not in exempt_commands:
        argv.insert(0, "document")

    app(args=argv, prog_name="pydocgen")
