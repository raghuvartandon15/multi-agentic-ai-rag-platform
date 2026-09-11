import re

def redact_credit_cards(text: str) -> str:
    """
    Detect common credit card number formats and redact them.
    Handles numbers with spaces, hyphens, or no separators.
    """
    pattern = r"\b(?:\d[ -]*?){13,19}\b"
    return re.sub(
        pattern,
        "[REDACTED CREDIT CARD]",
        text
    )

def redact_api_keys(text: str) -> str:
    """
    Basic POC detection for common API key formats.
    """
    patterns = [
        r"\bsk-[A-Za-z0-9_-]+\b",
        r"\bAKIA[A-Z0-9]{16}\b",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "[REDACTED SECRET]",
            text
        )

    return text

def apply_output_guardrails(text: str) -> str:
    """
    Apply all deterministic output guardrails.
    """

    text = redact_credit_cards(text)
    text = redact_api_keys(text)

    return text