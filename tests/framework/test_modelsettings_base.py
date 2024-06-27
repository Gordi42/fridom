import numpy as np
from fridom.framework import ModelSettingsBase

def test_doc_example(backend):
    class MyModelSettings(ModelSettingsBase):
        def __init__(self, grid, **kwargs):
            super().__init__(grid)
            self.model_name = "MyModel"
            self.my_parameter = 1.0
            self.set_attributes(**kwargs)
        def __str__(self) -> str:
            res = super().__str__()
            res += "  My parameter: {}\\n".format(self.my_parameter)
            return res

    mset = MyModelSettings(None, my_parameter=2.0)
    assert mset.model_name == "MyModel"
    assert mset.my_parameter == 2.0
    assert "My parameter: 2" in str(mset)