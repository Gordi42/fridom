"""Test the counter module."""
import pytest

import fridom.framework as fr


@pytest.fixture
def mset():
    return fr.ModelSettingsBase(
        grid=fr.grid.cartesian.Grid(shape=(16, ), domain_size=(1, )))

def test_defaut_counter(mset):
    number_of_steps = 100
    counter = fr.modules.Counter()
    counter.setup(mset)
    mz = fr.ModelState(mset)
    for _ in range(number_of_steps):
        mz = counter.update(mz)
        mz.clock.tick(1.0)
    assert counter.counter == number_of_steps

@pytest.mark.parametrize(*(
    "number_of_steps, step_size, expected_counter",
    [(100, 10, 10), (5, 2, 3)],
))
def test_counter_clock_trigger(number_of_steps, step_size,
                               expected_counter, mset):
    counter = fr.modules.Counter(fr.ClockTrigger(step_size=step_size))
    counter.setup(mset)
    mz = fr.ModelState(mset)
    for _ in range(number_of_steps):
        mz = counter.update(mz)
        mz.clock.tick(1.0)
    assert counter.counter == expected_counter
