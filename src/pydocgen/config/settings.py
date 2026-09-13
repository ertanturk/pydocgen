from typing import Final

# Configuration settings for the pydocgen service
SERVICE_NAME: Final[str] = "pydocgen"
ACCOUNT_NAME: Final[str] = "gemini-api-key"
ENV_API_KEY_NAMES: Final[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
MIN_KEY_LENGTH: Final[int] = 20
