from fridom.framework import config
from contextlib import contextmanager

class TimingComponent:
    """
    A class to keep track of the time spent in a particular component 
    of the model.

    Methods:
        start()    : start the timer
        stop()     : stop the timer
        reset()    : reset the timer
        __str__()  : return a string representation of the time spent
    """

    def __init__(self, name:str) -> None:
        """
        Initialize the TimingComponent.

        Arguments:
            name (str): name of the component
        """
        self.name = name
        self.time = 0.0
        self.is_active = False
        self.start_time = 0.0
        return

    def start(self) -> None:
        """
        Start the timer.
        """
        # check if the timer is already active
        if self.is_active:
            config.logger.warning(
                f"Start of TimingComponent {self.name} is called, but the component is already active.")
            return
        # start the timer
        self.is_active = True
        from time import time
        self.start_time = time()
        return

    def stop(self) -> None:
        """
        Stop the timer.
        """
        # check if the timer is active
        if not self.is_active:
            config.logger.warning(
                f"Stop of TimingComponent {self.name} is called, but the component is not active.")
            return
        # stop the timer
        from time import time
        self.time += time() - self.start_time
        self.is_active = False
        return
    
    def reset(self) -> None:
        """
        Reset the timer.
        """
        self.time = 0.0
        self.is_active = False
        self.start_time = 0.0
        return
    
    def __str__(self) -> str:
        """
        String representation of the time spent in the component.
        Format: "name: {hh:mm:ss}s"
        """
        # Convert time to hours, minutes, seconds
        t = self.time
        h = int(t // 3600)
        t -= h * 3600
        m = int(t // 60)
        t -= m * 60
        s = int(t)
        # print time in hours, minutes, seconds
        return f"{self.name:<30s}: {h:02d}:{m:02d}:{s:02d}s"

    
class TimingModule:
    """
    Container class for TimingComponent objects.

    Methods:
        add_component(name) : add a new TimingComponent
        get(name)           : return the TimingComponent with the given name
        reset()             : reset all TimingComponents
        __str__()           : return a string representation of the time spent
        __repr__()          : string representation for IPython
    """
    def __init__(self) -> None:
        """
        Construct the TimingModule with default components.
        """
        # initialize default components
        self.total               = TimingComponent("Total Integration")

        # add components to list
        self.components = [
            self.total,
        ]
        return

    def add_component(self, name:str) -> None:
        """
        Add a new TimingComponent to the TimingModule.

        Arguments:
            name (str): name of the new component
        """
        self.components.append(TimingComponent(name))
        return
    
    def get(self, name:str) -> TimingComponent:
        """
        Get the TimingComponent with the given name.
        If the component is not found, add a new component with the given name.

        Arguments:
            name (str): name of the component to get
        """
        # search for component with given name
        for component in self.components:
            if component.name == name:
                return component
        # if not found => add the component
        self.add_component(name)
        return self.get(name) # recursive call (should find the component now)

    def reset(self) -> None:
        """
        Reset all TimingComponents.
        """
        for component in self.components:
            component.reset()
        return

    @contextmanager
    def __getitem__(self, name:str):
        """
        Context manager to start and stop the timer for a component.

        Arguments:
            name (str): name of the component
        """
        component = self.get(name)
        component.start()
        yield component
        component.stop()
        return

    
    def __str__(self) -> str:
        """
        Return string representation of the model settings.
        """
        res = "=====================================================\n"
        res += " Timing Summary: \n"
        res += "=====================================================\n"
        for component in self.components:
            res += str(component)
            res += "   ({:.1f}%)\n".format(100 * component.time / self.total.time)
        res += "=====================================================\n"
        return res

    
    def __repr__(self) -> str:
        """
        Return string representation of the model settings (for IPython).
        """
        return self.__str__()