"""Heuristic warning for materially larger full sends with flat input usage.

Only fresh-context full sends on the same model/effort route are comparable.
Worker deltas measure increment characters against whole-context usage and are
excluded. Concurrent oneshots do not contribute samples. Cached input is included
conservatively, without changing usage reported to Hermes.

This is not proof of delivery: changed material density can cause false positives.
The first send, stable-size truncation, and ordinary tool loops are blind spots.
It does not prevent or repair soyelmismo/hermes-antigravity-subscription#29.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["DeliverySample", "DeliveryWatch", "ShortDelivery", "DISABLE_ENV"]

# Ignore small prompts where relative growth is noisy. This is not a safety limit.
MIN_PROMPT_CHARS = 20_000
# A jump smaller than this is ordinary turn-to-turn churn, not a rewrite.
MIN_CHARS_GROWTH = 8_000
# The prompt must have grown by at least this fraction ...
CHARS_GROWTH_RATIO = 0.25
# ... while delivered input tokens grew by no more than this fraction.
TOKEN_GROWTH_RATIO = 0.05

DISABLE_ENV = "ANTIGRAVITY_DISABLE_DELIVERY_WATCH"


@dataclass(frozen=True)
class DeliverySample:
    """One completed turn: what we assembled, and what the worker billed for."""

    prompt_chars: int
    input_tokens: int


@dataclass(frozen=True)
class ShortDelivery:
    """A prompt that grew materially while its delivered tokens did not."""

    sample: DeliverySample
    previous: DeliverySample

    @property
    def chars_growth(self) -> float:
        return (self.sample.prompt_chars - self.previous.prompt_chars) / max(
            self.previous.prompt_chars, 1
        )

    @property
    def token_growth(self) -> float:
        return (self.sample.input_tokens - self.previous.input_tokens) / max(
            self.previous.input_tokens, 1
        )

    def describe(self) -> str:
        """One line for the log: both sides of the comparison, in full."""
        return (
            "prompt sent to agy grew "
            f"{self.previous.prompt_chars:,} -> {self.sample.prompt_chars:,} chars "
            f"(+{self.chars_growth:.0%}) while the worker reported input tokens "
            f"{self.previous.input_tokens:,} -> {self.sample.input_tokens:,} "
            f"({self.token_growth:+.0%}); the prompt is probably being delivered "
            "shortened from the tail. Route long prompts through a chunked send "
            "or a provider that delivers whole prompts "
            "(soyelmismo/hermes-antigravity-subscription#29)."
        )


class DeliveryWatch:
    """Thread-safe record of valid, comparable full-send samples.

    A smaller send or route change clears the warning latch. Missing usage
    leaves the previous valid sample intact.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: DeliverySample | None = None
        self._sample_key = ""
        self._latched = False

    @property
    def last_sample(self) -> DeliverySample | None:
        with self._lock:
            return self._last

    def observe(
        self, prompt_chars: int, input_tokens: int, *, sample_key: str = ""
    ) -> ShortDelivery | None:
        """Compare valid full sends on the same route. This is only a heuristic."""
        if os.environ.get(DISABLE_ENV) or prompt_chars <= 0 or input_tokens <= 0:
            return None
        with self._lock:
            return self._observe_locked(prompt_chars, input_tokens, sample_key)

    def _observe_locked(self, prompt_chars: int, input_tokens: int, sample_key: str) -> ShortDelivery | None:
        current = DeliverySample(prompt_chars=int(prompt_chars), input_tokens=int(input_tokens))
        if sample_key != self._sample_key:
            self._last = None
            self._latched = False
            self._sample_key = sample_key
        previous, self._last = self._last, current

        # A shrunken prompt is a new conversation, a cleared context, or a
        # fresh chunked send: whatever it is, the previous sample no longer
        # describes this turn, and the latch should be free to fire again.
        if previous is None or current.prompt_chars < previous.prompt_chars:
            self._latched = False
            return None
        if current.prompt_chars < MIN_PROMPT_CHARS or current.input_tokens <= 0:
            return None
        if previous.input_tokens <= 0:
            return None
        if current.prompt_chars - previous.prompt_chars < MIN_CHARS_GROWTH:
            return None
        if self._latched:
            return None
        finding = ShortDelivery(sample=current, previous=previous)
        if finding.chars_growth < CHARS_GROWTH_RATIO:
            return None
        if finding.token_growth > TOKEN_GROWTH_RATIO:
            return None
        self._latched = True
        logger.warning("Antigravity prompt delivery looks short: %s", finding.describe())
        return finding
