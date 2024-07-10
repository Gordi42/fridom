import pytest
import fridom.framework as fr


# create a module that increments a number
class Increment(fr.modules.Module):
    def __init__(self):
        # sets the name of the module to "Increment", and the number to None
        super().__init__("Increment", number=None)
    
    @fr.modules.start_module
    def start(self):
        self.number = 0  # sets the number to 0

    @fr.modules.update_module
    def update(self, mz: fr.ModelSettingsBase, dz: fr.StateBase) -> None:
        self.number += 1  # increments the number by 1

    @fr.modules.stop_module
    def stop(self):
        self.number = None  # sets the number to None

@pytest.fixture()
def mset(backend):
    grid = fr.grid.CartesianGrid(N=(32, ), L=(1.0, ))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture()
def timer():
    return fr.timing_module.TimingModule()

def test_init(mset, timer):
    # create a module
    module = Increment()
    # check if the module is enabled
    assert module.is_enabled()
    # check if the module has the correct name
    assert module.name == "Increment"
    # check if the module has the correct number
    assert module.number == None
    # check if the module has the attributes grid, mset, and timer
    assert module.grid == None
    assert module.mset == None
    assert module.timer == None

def test_start(mset, timer):
    # create a module
    module = Increment()
    assert module.number == None
    # start the module
    module.start(mset=mset, timer=timer)
    # check if the number is 0
    assert module.number == 0
    # check if the grid, mset, and timer are set
    assert module.mset == mset
    assert module.grid == mset.grid
    assert module.timer == timer

    # now testing a disabled module
    module = Increment()
    module.disable()
    module.start(mset=mset, timer=timer)
    # check if everything is still None
    assert module.number == None
    assert module.grid == None
    assert module.mset == None
    assert module.timer == None

def test_update(mset, timer):
    # create a module
    module = Increment()
    # start the module
    module.start(mset=mset, timer=timer)
    # update the module
    module.update(mz=None, dz=None)
    # check if the number is 1
    assert module.number == 1
    # update the module again
    module.update(mz=None, dz=None)
    # check if the number is 2
    assert module.number == 2
    # disable the module
    module.disable()
    # update the module
    module.update(mz=None, dz=None)
    # check if the number is still 2
    assert module.number == 2
    
def test_stop(mset, timer):
    # create a module
    module = Increment()
    # start the module
    module.start(mset=mset, timer=timer)
    # check if the number is 0
    assert module.number == 0
    # stop the module
    module.stop()
    # check if the number is None
    assert module.number == None

def test_reset(mset, timer):
    # create a module
    module = Increment()
    # start the module
    module.start(mset=mset, timer=timer)
    module.update(mz=None, dz=None)
    # check if the number is 1
    assert module.number == 1
    # reset the module
    module.reset()
    # check if the number is None
    assert module.number == 0
