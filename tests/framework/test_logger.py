"""Tests for the logger module."""
import logging

import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def capture_logs():
    """Fixture to capture log output."""
    from io import StringIO

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    fr.log.addHandler(handler)
    yield stream
    fr.log.removeHandler(handler)

# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize(*(
    "level, value", [
        ("DEBUG", 10),
        ("INFO", 20),
        ("VERBOSE", 15),
        ("NOTICE", 25),
        ("WARNING", 30),
        ("ERROR", 40),
        ("CRITICAL", 50),
    ],
))
def test_log_level(capture_logs, level, value):
    """Test verbose log level."""
    # Set the log level
    fr.log.setLevel(level)

    # Log messages on all levels
    fr.log.debug("Debug message")
    fr.log.info("Info message")
    fr.log.verbose("Verbose message")
    fr.log.notice("Notice message")
    fr.log.warning("Warning message")
    fr.log.error("Error message")
    fr.log.critical("Critical message")

    logs = capture_logs.getvalue()
    messages = {"Debug message": 10,
                "Info message": 20,
                "Verbose message": 15,
                "Notice message": 25,
                "Warning message": 30,
                "Error message": 40,
                "Critical message": 50}

    # Check if the messages are in the log
    for message, log_level in messages.items():
        if log_level >= value:
            assert message in logs
        else:
            assert message not in logs
