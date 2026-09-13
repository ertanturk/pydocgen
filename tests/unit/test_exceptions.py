"""Unit tests for custom exceptions."""

import pytest

from pydocgen.errors.exceptions import (
    AnalysisError,
    ConfigurationError,
    CredentialAlreadyExistsError,
    CredentialError,
    CredentialNotFoundError,
    CredentialStorageError,
    GenerationError,
    InvalidCredentialError,
    ProviderAuthenticationError,
    ProviderError,
    PyDocGenError,
)


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        assert issubclass(PyDocGenError, Exception)

    def test_credential_exceptions_subclass_credential_error(self) -> None:
        assert issubclass(CredentialError, PyDocGenError)
        assert issubclass(InvalidCredentialError, CredentialError)
        assert issubclass(InvalidCredentialError, ValueError)
        assert issubclass(CredentialNotFoundError, CredentialError)
        assert issubclass(CredentialAlreadyExistsError, CredentialError)
        assert issubclass(CredentialStorageError, CredentialError)

    def test_domain_exceptions_subclass_pydocgen_error(self) -> None:
        assert issubclass(ConfigurationError, PyDocGenError)
        assert issubclass(ProviderError, PyDocGenError)
        assert issubclass(ProviderAuthenticationError, ProviderError)
        assert issubclass(AnalysisError, PyDocGenError)
        assert issubclass(GenerationError, PyDocGenError)

    def test_raising_invalid_credential_error(self) -> None:
        with pytest.raises(ValueError):
            raise InvalidCredentialError("Invalid key")

        with pytest.raises(CredentialError):
            raise InvalidCredentialError("Invalid key")

    def test_reexport_from_errors_package(self) -> None:
        import pydocgen.errors as err

        assert err.PyDocGenError is PyDocGenError
        assert err.CredentialError is CredentialError
        assert err.InvalidCredentialError is InvalidCredentialError
        assert err.CredentialNotFoundError is CredentialNotFoundError
        assert err.CredentialAlreadyExistsError is CredentialAlreadyExistsError
        assert err.CredentialStorageError is CredentialStorageError
