"""Tests for _sigterm_as_interrupt (defined in stilt.services)."""

import signal

import pytest

from stilt.executors import _sigterm_as_interrupt


def test_sigterm_as_interrupt_restores_handler():
    """Original SIGTERM handler is restored after the context exits."""
    original = signal.getsignal(signal.SIGTERM)
    with _sigterm_as_interrupt():
        assert signal.getsignal(signal.SIGTERM) != original
    assert signal.getsignal(signal.SIGTERM) == original


def test_sigterm_as_interrupt_raises_keyboard_interrupt():
    """SIGTERM delivered inside the context raises KeyboardInterrupt."""
    with pytest.raises(KeyboardInterrupt), _sigterm_as_interrupt():
        signal.raise_signal(signal.SIGTERM)


def test_sigterm_as_interrupt_restores_after_exception():
    """Handler is restored even when an exception interrupts the block."""
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), _sigterm_as_interrupt():
        signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == original
