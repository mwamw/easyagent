"""Public provider-neutral conversation data and conversion helpers."""

from core.history import CanonicalMessage, canonical_text_content, coerce_canonical_message

__all__ = ["CanonicalMessage", "canonical_text_content", "coerce_canonical_message"]
