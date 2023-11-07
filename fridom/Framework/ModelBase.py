from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase
from fridom.Framework.StateBase import StateBase
from fridom.Framework.TimingModule import TimingModule


class ModelBase:
    """
    Base class for the model.

    Methods:
        step()                  : Perform one time step.
        run()                   : Run the model for a given number of steps.
        reset()                 : Reset the model (pointers, tendencies)
        nonlinear_dz()          : Calculate nonlinear tendency.
    """

    def __init__(self, mset:ModelSettingsBase, grid:GridBase) -> None:
        """
        Constructor.

        Args:
            mset (ModelSettings)    : Model settings.
            grid (Grid)             : Grid.
        """
        self.mset = mset
        self.grid = grid

        # Timer
        self.timer = TimingModule()

        # Iteration counter
        self.it = 0
        return


    # ============================================================
    #   RUN MODEL
    # ============================================================

    def run(self, steps=None, runlen=None) -> None:
        """
        Run the model for a given number of steps or a given time.

        Args:
            steps (int)     : Number of steps to run.
            runlen (float)  : Time to run. (preferred over steps)
        """
        # to implement in child class
        return

    
    # ============================================================
    #   SINGLE TIME STEP
    # ============================================================

    def step(self) -> None:
        """
        Update the model state by one time step.
        """
        self.it += 1
        # to implement in child class
        return


    # ============================================================
    #   TENDENCIES
    # ============================================================

    def nonlinear_dz(self) -> StateBase:
        """
        Calculate nonlinear tendency.

        Returns:
            dz (State)  : Nonlinear tendency.
        """
        # to implement in child class
        return

    # ============================================================
    #   OTHER METHODS
    # ============================================================

    def reset(self) -> None:
        """
        Reset the model (pointers, tendencies).
        """
        self.it = 0
        self.timer.reset()
        # to implement in child class
        return

    @property
    def time(self) -> float:
        """
        Model time.
        """
        return self.it * self.mset.dt