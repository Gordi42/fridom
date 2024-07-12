import pytest
import fridom.framework as fr


# create a module that increments a number
class Increment(fr.modules.Module):
    def __init__(self):
        # sets the name of the module to "Increment", and the number to None
        super().__init__("Increment", number=None)
    
    @fr.modules.setup_module
    def setup(self):
        self.number = 0  # sets the number to 0

    @fr.modules.module_method
    def start(self):
        self.number = 0  # sets the number to 0

    @fr.modules.module_method
    def update(self, mz: fr.ModelSettingsBase) -> None:
        self.number += 1  # increments the number by 1

    @fr.modules.module_method
    def stop(self):
        self.number = None  # sets the number to None

@pytest.fixture()
def mset(backend):
    grid = fr.grid.CartesianGrid(N=(32, ), L=(1.0, ))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

def test_init(mset):
    # create a module
    module = Increment()
    # check if the module is enabled
    assert module.is_enabled()
    # check if the module has the correct name
    assert module.name == "Increment"
    # check if the module has the correct number
    assert module.number == None
    # check if the module has the attributes grid, mset, and timer
    assert module.mset == None
    assert module.timer == None

def test_start(mset):
    # create a module
    module = Increment()
    assert module.number == None
    # start the module
    module.setup(mset=mset)
    # check if the number is 0
    assert module.number == 0
    # check if the grid, mset, and timer are set
    assert module.mset == mset

    # now testing a disabled module
    module = Increment()
    module.disable()
    module.setup(mset=mset)
    # check if everything is still None
    assert module.number == None
    assert module.mset == None

def test_update(mset):
    # create a module
    module = Increment()
    # start the module
    module.setup(mset=mset)
    # update the module
    module.update(mz=None)
    # check if the number is 1
    assert module.number == 1
    # update the module again
    module.update(mz=None)
    # check if the number is 2
    assert module.number == 2
    # disable the module
    module.disable()
    # update the module
    module.update(mz=None)
    # check if the number is still 2
    assert module.number == 2
    
def test_stop(mset):
    # create a module
    module = Increment()
    # start the module
    module.setup(mset=mset)
    # check if the number is 0
    assert module.number == 0
    # stop the module
    module.stop()
    # check if the number is None
    assert module.number == None

def test_reset(mset):
    # create a module
    module = Increment()
    # start the module
    module.setup(mset=mset)
    module.update(mz=None)
    # check if the number is 1
    assert module.number == 1
    # reset the module
    module.reset()
    # check if the number is None
    assert module.number == 0
