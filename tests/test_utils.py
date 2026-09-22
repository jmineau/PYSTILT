"""Tests for sigterm_as_interrupt."""

import signal
import threading

import pytest

from stilt.execution import sigterm_as_interrupt


def test_sigterm_as_interrupt_restores_handler():
    """Original SIGTERM handler is restored after the context exits."""
    original = signal.getsignal(signal.SIGTERM)
    with sigterm_as_interrupt():
        assert signal.getsignal(signal.SIGTERM) != original
    assert signal.getsignal(signal.SIGTERM) == original


def test_sigterm_as_interrupt_raises_keyboard_interrupt():
    """SIGTERM delivered inside the context raises KeyboardInterrupt."""
    with pytest.raises(KeyboardInterrupt), sigterm_as_interrupt():
        signal.raise_signal(signal.SIGTERM)


def test_sigterm_as_interrupt_restores_after_exception():
    """Handler is restored even when an exception interrupts the block."""
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), sigterm_as_interrupt():
        signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == original


def test_sigterm_as_interrupt_is_noop_outside_main_thread():
    """Called from a non-main thread, the SIGTERM handler is left untouched."""
    original = signal.getsignal(signal.SIGTERM)
    seen: dict[str, object] = {}

    def _worker() -> None:
        with sigterm_as_interrupt():
            seen["inside"] = signal.getsignal(signal.SIGTERM)
        seen["after"] = signal.getsignal(signal.SIGTERM)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert seen["inside"] == original
    assert seen["after"] == original
    assert signal.getsignal(signal.SIGTERM) == original
