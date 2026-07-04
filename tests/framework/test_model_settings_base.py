"""Tests for the ModelSettingsBase class."""

import fridom.framework as fr

# ================================================================
#  Fixtures
# ================================================================


# ================================================================
#  Tests
# ================================================================

def test_init(): ...

def test_attributes(): ...

def test_setup(): ...

def test_repr(): ...

def test_halo(): ...

def test_doc_example():
    class MyModelSettings(fr.ModelSettingsBase):
        def __init__(self, grid, **kwargs):
            super().__init__(grid)
            self.model_name = "MyModel"
            self.my_parameter = 1.0
            self.set_attributes(**kwargs)
        def __str__(self) -> str:
            res = super().__str__()
            res += f"  My parameter: {self.my_parameter}\\n"
            return res

    mset = MyModelSettings(None, my_parameter=2.0)
    assert mset.model_name == "MyModel"
    assert mset.my_parameter == 2.0
    assert "My parameter: 2" in str(mset)
