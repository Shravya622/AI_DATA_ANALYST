"""
Unit tests for app/exceptions.py — verifies that each custom exception
carries a human-readable ``message`` attribute and is importable cleanly.
"""

import pytest

from app.exceptions import (
    ConfigError,
    DatasetNotFoundError,
    SessionNotFoundError,
    UnknownToolError,
)

# ---------------------------------------------------------------------------
# Parametrise over all four exception classes for DRY coverage
# ---------------------------------------------------------------------------

EXCEPTION_CLASSES = [
    SessionNotFoundError,
    DatasetNotFoundError,
    UnknownToolError,
    ConfigError,
]


@pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
def test_message_attribute_stored(exc_class):
    """Each exception must store the message on self.message."""
    msg = f"test message for {exc_class.__name__}"
    exc = exc_class(msg)
    assert exc.message == msg


@pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
def test_message_in_args(exc_class):
    """Message must also appear in args (via super().__init__) so str(exc) works."""
    msg = f"args message for {exc_class.__name__}"
    exc = exc_class(msg)
    assert msg in str(exc)
    assert exc.args[0] == msg


@pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
def test_is_subclass_of_exception(exc_class):
    """Each custom exception must be a subclass of the built-in Exception."""
    assert issubclass(exc_class, Exception)


@pytest.mark.parametrize("exc_class", EXCEPTION_CLASSES)
def test_can_be_raised_and_caught(exc_class):
    """Exceptions must be raise-able and catch-able like normal exceptions."""
    msg = "raised message"
    with pytest.raises(exc_class) as exc_info:
        raise exc_class(msg)
    assert exc_info.value.message == msg


def test_session_not_found_distinct_from_dataset_not_found():
    """The four exceptions must be distinct types (not aliases of each other)."""
    assert SessionNotFoundError is not DatasetNotFoundError
    assert SessionNotFoundError is not UnknownToolError
    assert SessionNotFoundError is not ConfigError
    assert DatasetNotFoundError is not UnknownToolError
    assert DatasetNotFoundError is not ConfigError
    assert UnknownToolError is not ConfigError
