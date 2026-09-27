import pytest

from rx590gme import RX590GME


@pytest.fixture
def gpu():
    return RX590GME(watchdog_instructions=200_000)
