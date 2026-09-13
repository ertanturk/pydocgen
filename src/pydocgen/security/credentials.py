"""Credentials management for the pydocgen service."""

import os

import keyring
import keyring.errors

from pydocgen.config.settings import (
    ACCOUNT_NAME,
    ENV_API_KEY_NAMES,
    MIN_KEY_LENGTH,
    SERVICE_NAME,
)
from pydocgen.errors.exceptions import (
    CredentialAlreadyExistsError,
    CredentialNotFoundError,
    CredentialStorageError,
    InvalidCredentialError,
)


class Credentials:
    """Secure credentials manager using system keyring with environment variable fallback."""

    service_name: str = SERVICE_NAME
    account_name: str = ACCOUNT_NAME

    @classmethod
    def _validate_api_key(cls, api_key: str) -> str:
        """Validate and sanitize an API key.

        Args:
            api_key: The API key string to validate.

        Returns:
            The stripped, validated API key.

        Raises:
            InvalidCredentialError: If the API key is not a string, empty, contains
                control characters, or does not meet length requirements.
        """
        if not isinstance(api_key, str):
            raise InvalidCredentialError("API key must be a string.")

        sanitized_key = api_key.strip()
        if not sanitized_key:
            raise InvalidCredentialError("API key cannot be empty or contain only whitespace.")

        if any(c in sanitized_key for c in ("\n", "\r", "\t", "\0")):
            raise InvalidCredentialError(
                "API key contains invalid control characters (e.g., newlines, tabs, or null bytes)."
            )

        if not sanitized_key.isascii() or not all(c.isprintable() for c in sanitized_key):
            raise InvalidCredentialError("API key must contain only printable ASCII characters.")

        if len(sanitized_key) < MIN_KEY_LENGTH:
            raise InvalidCredentialError(
                f"API key is too short (minimum {MIN_KEY_LENGTH} characters required)."
            )

        return sanitized_key

    @classmethod
    def mask_api_key(cls, api_key: str | None) -> str:
        """Mask an API key for safe display and logging without revealing secrets.

        Args:
            api_key: The API key string to mask.

        Returns:
            Masked string (e.g., 'AIza...1234') or empty string if input is falsy.
        """
        if not api_key:
            return ""

        sanitized = api_key.strip()

        return f"{sanitized[:4]}...{sanitized[-4:]}"

    @classmethod
    def save_api_key(cls, api_key: str, *, overwrite: bool = False) -> None:
        """Save an API key to secure keyring storage.

        Args:
            api_key: The API key to store.
            overwrite: If True, overwrite an existing stored key instead of raising an error.

        Raises:
            InvalidCredentialError: If the API key fails validation.
            CredentialAlreadyExistsError: If a key already exists in keyring and overwrite is False.
            CredentialStorageError: If the keyring backend encounters an error.
        """
        validated_key = cls._validate_api_key(api_key)

        if not overwrite and cls.has_api_key(include_env=False):
            raise CredentialAlreadyExistsError(
                "API key already exists in keyring. Use update_api_key() or pass overwrite=True."
            )

        try:
            keyring.set_password(cls.service_name, cls.account_name, validated_key)
        except keyring.errors.KeyringError as e:
            raise CredentialStorageError(
                f"Failed to save API key to secure keyring storage: {e}"
            ) from e
        except Exception as e:
            raise CredentialStorageError(
                f"Unexpected error occurred while saving API key to storage: {e}"
            ) from e

    @classmethod
    def get_api_key(cls, *, include_env: bool = True) -> str | None:
        """Retrieve the API key from secure keyring storage or environment variables.

        Args:
            include_env: Whether to check environment variables (GEMINI_API_KEY,
                GOOGLE_API_KEY) if no key is stored in the keyring.

        Returns:
            The stored API key, or None if not found.

        Raises:
            CredentialStorageError: If the keyring backend encounters an error and
                no environment fallback is available.
        """
        keyring_error: Exception | None = None

        try:
            stored_key = keyring.get_password(cls.service_name, cls.account_name)
            if stored_key and stored_key.strip():
                return stored_key.strip()
        except (keyring.errors.KeyringError, Exception) as e:
            keyring_error = e

        if include_env:
            for env_var in ENV_API_KEY_NAMES:
                env_key = os.environ.get(env_var)
                if env_key and env_key.strip():
                    return env_key.strip()

        if keyring_error is not None:
            raise CredentialStorageError(
                f"Failed to access secure keyring storage: {keyring_error}"
            ) from keyring_error

        return None

    @classmethod
    def update_api_key(cls, api_key: str) -> None:
        """Update an existing API key in secure keyring storage.

        Args:
            api_key: The new API key.

        Raises:
            InvalidCredentialError: If the API key fails validation.
            CredentialNotFoundError: If no existing key is found in keyring to update.
            CredentialStorageError: If the keyring backend encounters an error.
        """
        validated_key = cls._validate_api_key(api_key)

        if not cls.has_api_key(include_env=False):
            raise CredentialNotFoundError(
                "No existing API key found in keyring to update. Use save_api_key() instead."
            )

        try:
            keyring.set_password(cls.service_name, cls.account_name, validated_key)
        except keyring.errors.KeyringError as e:
            raise CredentialStorageError(
                f"Failed to update API key in secure keyring storage: {e}"
            ) from e
        except Exception as e:
            raise CredentialStorageError(
                f"Unexpected error occurred while updating API key: {e}"
            ) from e

    @classmethod
    def delete_api_key(cls) -> None:
        """Delete the API key from secure keyring storage.

        Raises:
            CredentialNotFoundError: If no API key exists in keyring to delete.
            CredentialStorageError: If the keyring backend encounters an error.
        """
        if not cls.has_api_key(include_env=False):
            raise CredentialNotFoundError("No API key found in keyring to delete.")

        try:
            keyring.delete_password(cls.service_name, cls.account_name)
        except keyring.errors.PasswordDeleteError as e:
            raise CredentialNotFoundError("No API key found in keyring to delete.") from e
        except keyring.errors.KeyringError as e:
            raise CredentialStorageError(
                f"Failed to delete API key from secure keyring storage: {e}"
            ) from e
        except Exception as e:
            raise CredentialStorageError(
                f"Unexpected error occurred while deleting API key: {e}"
            ) from e

    @classmethod
    def has_api_key(cls, *, include_env: bool = True) -> bool:
        """Check whether an API key exists in storage or environment.

        Args:
            include_env: Whether to check environment variables if not found in keyring.

        Returns:
            True if a valid, non-empty API key is present, False otherwise.

        Raises:
            CredentialStorageError: If the keyring backend fails and include_env is False
                or no environment fallback is available.
        """
        return cls.get_api_key(include_env=include_env) is not None
