from typing import Final

# Configuration settings for the pydocgen service
SERVICE_NAME: Final[str] = "pydocgen"
ACCOUNT_NAME: Final[str] = "gemini-api-key"
ENV_API_KEY_NAMES: Final[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
MIN_KEY_LENGTH: Final[int] = 20
# 1 token ≈ 3.5 characters for code constructs (operators, indentation, identifiers)
CHARS_PER_TOKEN = 3.5
# Fixed estimate for system prompt, schema constraints, and format instructions
DEFAULT_PROMPT_OVERHEAD_TOKENS = 500
# Structural JSON framing per function (keys, quotes, function_id wrapper)
PER_FUNCTION_OVERHEAD_TOKENS = 40
