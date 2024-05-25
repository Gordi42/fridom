import unittest
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

class ModuleTest(unittest.TestCase):
    """
    Test the module class.
    """
    def setUp(self) -> None:
        # create a grid and timer
        mset = fr.ModelSettingsBase(n_dims=1)
        grid = fr.GridBase(mset)
        timer = fr.timing_module.TimingModule()
        # set the grid and timer as attributes of the testing module
        self.mset = mset
        self.grid = grid
        self.timer = timer
        return

    def test_init(self):
        """
        Test the initialization of the module.
        """
        # create a module
        module = Increment()
        # check if the module is enabled
        self.assertTrue(module.is_enabled())
        # check if the module has the correct name
        self.assertEqual(module.name, "Increment")
        # check if the module has the correct number
        self.assertEqual(module.number, None)
        # check if the module has the attributes grid, mset, and timer
        self.assertEqual(module.grid, None)
        self.assertEqual(module.mset, None)
        self.assertEqual(module.timer, None)
        return

    def test_start(self):
        """
        Test the start method of the module.
        """
        # create a module
        module = Increment()
        # start the module
        module.start(grid=self.grid, timer=self.timer)
        # check if the number is 0
        self.assertEqual(module.number, 0)
        # check if the grid is set
        self.assertEqual(module.grid, self.grid)
        # check if the model settings are set
        self.assertEqual(module.mset, self.mset)
        # check if the timer is set
        self.assertEqual(module.timer, self.timer)

        # now testing a disabled module
        module = Increment()
        module.disable()
        module.start(grid=self.grid, timer=self.timer)
        # check if the number is None and the grid, mset, and timer are not set
        self.assertEqual(module.number, None)
        self.assertEqual(module.grid, None)
        self.assertEqual(module.mset, None)
        self.assertEqual(module.timer, None)
        return

    def test_update(self):
        """
        Test the update method of the module.
        """
        # create a module
        module = Increment()
        # start the module
        module.start(grid=self.grid, timer=self.timer)
        # check if the number is 0
        self.assertEqual(module.number, 0)
        # update the module
        module.update(mz=None, dz=None)
        # check if the number is 1
        self.assertEqual(module.number, 1)
        # disable the module
        module.disable()
        # update the module
        module.update(mz=None, dz=None)
        # check if the number is still 1
        self.assertEqual(module.number, 1)
        return

    def test_stop(self):
        """
        Test the stop method of the module.
        """
        # create a module
        module = Increment()
        # start the module
        module.start(grid=self.grid, timer=self.timer)
        # check if the number is 0
        self.assertEqual(module.number, 0)
        # stop the module
        module.stop()
        # check if the number is None
        self.assertEqual(module.number, None)
        return

    def test_reset(self):
        """
        Test the reset method of the module.
        """
        # create a module
        module = Increment()
        # start the module
        module.start(grid=self.grid, timer=self.timer)
        # check if the number is 0
        self.assertEqual(module.number, 0)
        # update the module
        module.update(mz=None, dz=None)
        # check if the number is 1
        self.assertEqual(module.number, 1)
        # reset the module
        module.reset()
        # check if the number is 0
        self.assertEqual(module.number, 0)
        return
