"""
Secret redaction helpers.

Some data providers only accept API keys as URL query parameters, and
httpx/raise_for_status put full URLs into logs and exception text. These
helpers scrub keys before they reach logs or API responses.
"""

import logging
import re

from config.settings import settings

_QUERY_SECRET = re.compile(
    r"(?i)(\b(?:api[_-]?key|apikey|token|access_token|key)=)([^&\s\"'<>]+)"
)
_REDACTED = "***"


def _known_secrets() -> list[str]:
    values = [
        settings.anthropic_api_key,
        settings.openai_api_key,
        settings.fmp_api_key,
        settings.alpha_vantage_api_key,
        settings.finnhub_api_key,
        settings.polygon_api_key,
        settings.api_key,
    ]
    return [v for v in values if v and len(v) >= 6]


def redact_secrets(text) -> str:
    """Remove API keys from text (query-string keys and configured key values)."""
    text = _QUERY_SECRET.sub(lambda m: f"{m.group(1)}{_REDACTED}", str(text))
    for secret in _known_secrets():
        text = text.replace(secret, _REDACTED)
    return text


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record it sees."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = None
        if message is not None:
            redacted = redact_secrets(message)
            if redacted != message:
                record.msg = redacted
                record.args = ()
        # The formatter renders tracebacks separately from the message, so
        # exc_info (and any cached exc_text) needs redacting too.
        if record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        if record.exc_info:
            record.exc_text = redact_secrets(
                logging.Formatter().formatException(record.exc_info)
            )
            record.exc_info = None
        return True


def install_log_redaction() -> None:
    """Attach redaction to all current log handlers and quiet URL-logging libraries."""
    redactor = RedactingFilter()
    names = ["", "uvicorn", "uvicorn.error", "uvicorn.access"]
    for name in names:
        for handler in logging.getLogger(name).handlers:
            if not any(isinstance(f, RedactingFilter) for f in handler.filters):
                handler.addFilter(redactor)
    # httpx logs every request URL at INFO
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
