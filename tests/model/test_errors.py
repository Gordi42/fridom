"""Tests for the model-layer error registry (model/errors.py)."""
import pytest

from fridom.model import errors as model_errors
from fridom.spatial import errors as spatial_errors


# ================================================================
#  Re-exports from the grid cluster
# ================================================================
@pytest.mark.parametrize("name", [
    pytest.param("GridFrozenError", id="grid-frozen"),
    pytest.param("ImmutableStateError", id="immutable-state"),
])
def test_grid_cluster_errors_are_re_exported_not_redefined(name):
    # both classes live at their raise sites in the grid cluster;
    # model.errors must hand out the SAME object, not a parallel
    # class, so one `except` catches across both layers
    assert getattr(model_errors, name) is getattr(spatial_errors, name)
    assert name in model_errors.__all__
