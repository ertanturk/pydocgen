"""Unit tests for Credentials management and security."""

from unittest.mock import MagicMock, patch

import keyring.errors
import pytest

from pydocgen.errors.exceptions import (
    CredentialAlreadyExistsError,
    CredentialNotFoundError,
    CredentialStorageError,
    InvalidCredentialError,
)
from pydocgen.security.credentials import (
    ACCOUNT_NAME,
    SERVICE_NAME,
    Credentials,
)

VALID_API_KEY = "AIzaSyD-1234567890abcdefghijklmnop"


class TestCredentialsValidation:
    """Test input validation and sanitization for API keys."""

    def test_valid_api_key(self) -> None:
        result = Credentials._validate_api_key(VALID_API_KEY)
        assert result == VALID_API_KEY

    def test_valid_api_key_with_whitespace_stripping(self) -> None:
        result = Credentials._validate_api_key(f"  {VALID_API_KEY}  \n")
        # Notice: \n at the end gets stripped by .strip()
        assert result == VALID_API_KEY

    def test_rejects_non_string(self) -> None:
        for invalid in [None, 12345, ["key"], {"key": "val"}]:
            with pytest.raises(InvalidCredentialError, match="must be a string"):
                Credentials._validate_api_key(invalid)  # ty: ignore[invalid-argument-type]

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(InvalidCredentialError, match="cannot be empty"):
            Credentials._validate_api_key("")

    def test_rejects_whitespace_only_string(self) -> None:
        """Addresses the logic bug where '   ' bypassed 'if not api_key'."""
        with pytest.raises(
            InvalidCredentialError, match="cannot be empty or contain only whitespace"
        ):
            Credentials._validate_api_key("   \t  \n  ")

    def test_rejects_embedded_newlines(self) -> None:
        with pytest.raises(InvalidCredentialError, match="invalid control characters"):
            Credentials._validate_api_key("AIzaSy\n1234567890abcdef")

    def test_rejects_embedded_carriage_returns(self) -> None:
        with pytest.raises(InvalidCredentialError, match="invalid control characters"):
            Credentials._validate_api_key("AIzaSy\r1234567890abcdef")

    def test_rejects_embedded_tabs(self) -> None:
        with pytest.raises(InvalidCredentialError, match="invalid control characters"):
            Credentials._validate_api_key("AIzaSy\t1234567890abcdef")

    def test_rejects_embedded_null_byte(self) -> None:
        with pytest.raises(InvalidCredentialError, match="invalid control characters"):
            Credentials._validate_api_key("AIzaSy\01234567890abcdef")

    def test_rejects_non_ascii_characters(self) -> None:
        with pytest.raises(InvalidCredentialError, match="printable ASCII"):
            Credentials._validate_api_key("AIzaSy-Türkçe-12345")

    def test_rejects_too_short_key(self) -> None:
        with pytest.raises(InvalidCredentialError, match="too short"):
            Credentials._validate_api_key("short")


class TestMaskApiKey:
    """Test API key masking to ensure sensitive keys are not leaked."""

    def test_mask_empty_or_none(self) -> None:
        assert Credentials.mask_api_key("") == ""
        assert Credentials.mask_api_key(None) == ""

    def test_mask_standard_key(self) -> None:
        assert Credentials.mask_api_key(VALID_API_KEY) == "AIza...mnop"


class TestSaveApiKey:
    """Test save_api_key behavior, lifecycle checks, and error handling."""

    @patch("keyring.set_password")
    @patch.object(Credentials, "has_api_key", return_value=False)
    def test_save_new_key_success(self, mock_has_key: MagicMock, mock_set_pwd: MagicMock) -> None:
        Credentials.save_api_key(VALID_API_KEY)
        mock_has_key.assert_called_once_with(include_env=False)
        mock_set_pwd.assert_called_once_with(SERVICE_NAME, ACCOUNT_NAME, VALID_API_KEY)

    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_save_existing_key_raises_error(self, mock_has_key: MagicMock) -> None:
        with pytest.raises(CredentialAlreadyExistsError, match="already exists"):
            Credentials.save_api_key(VALID_API_KEY)

    @patch("keyring.set_password")
    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_save_existing_key_with_overwrite(
        self, mock_has_key: MagicMock, mock_set_pwd: MagicMock
    ) -> None:
        Credentials.save_api_key(VALID_API_KEY, overwrite=True)
        mock_set_pwd.assert_called_once_with(SERVICE_NAME, ACCOUNT_NAME, VALID_API_KEY)

    @patch("keyring.set_password", side_effect=keyring.errors.KeyringError("Backend locked"))
    @patch.object(Credentials, "has_api_key", return_value=False)
    def test_save_keyring_error_wrapped(
        self, mock_has_key: MagicMock, mock_set_pwd: MagicMock
    ) -> None:
        with pytest.raises(CredentialStorageError, match="Backend locked"):
            Credentials.save_api_key(VALID_API_KEY)


class TestGetApiKey:
    """Test get_api_key retrieval from keyring and environment variables."""

    @patch("keyring.get_password", return_value=VALID_API_KEY)
    def test_get_from_keyring_success(self, mock_get_pwd: MagicMock) -> None:
        key = Credentials.get_api_key()
        assert key == VALID_API_KEY
        mock_get_pwd.assert_called_once_with(SERVICE_NAME, ACCOUNT_NAME)

    @patch("keyring.get_password", return_value="   ")
    @patch.dict("os.environ", {}, clear=True)
    def test_get_whitespace_from_keyring_treated_as_none(self, mock_get_pwd: MagicMock) -> None:
        key = Credentials.get_api_key()
        assert key is None

    @patch("keyring.get_password", return_value=None)
    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-gemini-key-12345"}, clear=True)
    def test_fallback_to_gemini_env_var(self, mock_get_pwd: MagicMock) -> None:
        key = Credentials.get_api_key(include_env=True)
        assert key == "env-gemini-key-12345"

    @patch("keyring.get_password", return_value=None)
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-google-key-12345"}, clear=True)
    def test_fallback_to_google_env_var(self, mock_get_pwd: MagicMock) -> None:
        key = Credentials.get_api_key(include_env=True)
        assert key == "env-google-key-12345"

    @patch("keyring.get_password", return_value=None)
    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-gemini-key-12345"}, clear=True)
    def test_do_not_fallback_when_include_env_false(self, mock_get_pwd: MagicMock) -> None:
        key = Credentials.get_api_key(include_env=False)
        assert key is None

    @patch("keyring.get_password", side_effect=keyring.errors.KeyringError("D-Bus failure"))
    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-gemini-key-12345"}, clear=True)
    def test_keyring_error_falls_back_to_env(self, mock_get_pwd: MagicMock) -> None:
        key = Credentials.get_api_key(include_env=True)
        assert key == "env-gemini-key-12345"

    @patch("keyring.get_password", side_effect=keyring.errors.KeyringError("D-Bus failure"))
    @patch.dict("os.environ", {}, clear=True)
    def test_keyring_error_without_env_raises_storage_error(self, mock_get_pwd: MagicMock) -> None:
        with pytest.raises(CredentialStorageError, match="D-Bus failure"):
            Credentials.get_api_key(include_env=True)

    @patch("keyring.get_password", side_effect=keyring.errors.KeyringError("D-Bus failure"))
    def test_keyring_error_with_include_env_false_raises(self, mock_get_pwd: MagicMock) -> None:
        with pytest.raises(CredentialStorageError, match="D-Bus failure"):
            Credentials.get_api_key(include_env=False)


class TestUpdateApiKey:
    """Test update_api_key lifecycle check and error handling."""

    @patch("keyring.set_password")
    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_update_existing_key_success(
        self, mock_has_key: MagicMock, mock_set_pwd: MagicMock
    ) -> None:
        Credentials.update_api_key(VALID_API_KEY)
        mock_has_key.assert_called_once_with(include_env=False)
        mock_set_pwd.assert_called_once_with(SERVICE_NAME, ACCOUNT_NAME, VALID_API_KEY)

    @patch.object(Credentials, "has_api_key", return_value=False)
    def test_update_nonexistent_key_raises_not_found(self, mock_has_key: MagicMock) -> None:
        """Addresses the logic bug where update silently created keys even if none existed."""
        with pytest.raises(CredentialNotFoundError, match="No existing API key found"):
            Credentials.update_api_key(VALID_API_KEY)

    @patch("keyring.set_password", side_effect=keyring.errors.KeyringError("Storage full"))
    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_update_keyring_error_wrapped(
        self, mock_has_key: MagicMock, mock_set_pwd: MagicMock
    ) -> None:
        with pytest.raises(CredentialStorageError, match="Storage full"):
            Credentials.update_api_key(VALID_API_KEY)


class TestDeleteApiKey:
    """Test delete_api_key behavior and error handling."""

    @patch("keyring.delete_password")
    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_delete_existing_key_success(
        self, mock_has_key: MagicMock, mock_del_pwd: MagicMock
    ) -> None:
        Credentials.delete_api_key()
        mock_has_key.assert_called_once_with(include_env=False)
        mock_del_pwd.assert_called_once_with(SERVICE_NAME, ACCOUNT_NAME)

    @patch.object(Credentials, "has_api_key", return_value=False)
    def test_delete_nonexistent_key_raises_not_found(self, mock_has_key: MagicMock) -> None:
        with pytest.raises(CredentialNotFoundError, match="No API key found in keyring to delete"):
            Credentials.delete_api_key()

    @patch(
        "keyring.delete_password",
        side_effect=keyring.errors.PasswordDeleteError("Password not found"),
    )
    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_delete_password_delete_error_wrapped(
        self, mock_has_key: MagicMock, mock_del_pwd: MagicMock
    ) -> None:
        with pytest.raises(CredentialNotFoundError, match="No API key found"):
            Credentials.delete_api_key()

    @patch("keyring.delete_password", side_effect=keyring.errors.KeyringError("Backend broken"))
    @patch.object(Credentials, "has_api_key", return_value=True)
    def test_delete_keyring_error_wrapped(
        self, mock_has_key: MagicMock, mock_del_pwd: MagicMock
    ) -> None:
        with pytest.raises(CredentialStorageError, match="Backend broken"):
            Credentials.delete_api_key()


class TestHasApiKey:
    """Test has_api_key query method."""

    @patch.object(Credentials, "get_api_key", return_value=VALID_API_KEY)
    def test_has_api_key_true(self, mock_get_api_key: MagicMock) -> None:
        assert Credentials.has_api_key() is True

    @patch.object(Credentials, "get_api_key", return_value=None)
    def test_has_api_key_false(self, mock_get_api_key: MagicMock) -> None:
        assert Credentials.has_api_key() is False
