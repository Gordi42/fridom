import unittest
import numpy as np

import sys
sys.path.append('../..')

from fridom.framework.modelsettings_base import ModelSettingsBase
from fridom.framework.grid_base import GridBase
from fridom.framework.boundary_conditions import *

class TestPeriodicBoundary(unittest.TestCase):
    """
    Test the base class for boundary conditions.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """

        m = ModelSettingsBase(2)
        for i in range(2):
            b = Periodic(m, i)
            self.assertEqual(b.mset, m)
            self.assertEqual(b.axis, i)

    def test_pad_for_fft(self):
        """
        Test the padding for FFT.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)

        periods = [[True, True], [True, False], [False, True], [False, False]]

        for period in periods:
            mset.periodic_bounds = period
            grid = GridBase(mset)

            for i in range(n_dims):
                p = Periodic(mset, i)
                x = grid.X[i]
                x_pad = p.pad_for_fft(x)
                # x and x_pad should be the same
                self.assertTrue(np.allclose(x, x_pad))

    def test_pad_unpad(self):
        """
        Test if padding and unpadding are inverse operations.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)

        periods = [[True, True], [True, False], [False, True], [False, False]]

        for period in periods:
            mset.periodic_bounds = period
            grid = GridBase(mset)

            for i in range(n_dims):
                p = Periodic(mset, i)
                x = grid.X[i]
                x_pad = p.pad_for_fft(x)
                x_unpad = p.unpad_from_fft(x_pad)
                # x and x_unpad should be the same
                self.assertTrue(np.allclose(x, x_unpad))


class TestDirichlet(unittest.TestCase):
    """
    Test the Dirichlet boundary conditions.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """

        m = ModelSettingsBase(2)
        for i in range(2):
            b = Dirichlet(m, i, 1)
            self.assertEqual(b.mset, m)
            self.assertEqual(b.axis, i)

    def test_pad_shape_type1(self):
        """
        Test if the padding has the correct shape.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)

        for i in range(n_dims):
            p = Dirichlet(mset, i, btype=1)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            shape = list(x.shape)
            shape[i] *= 2
            self.assertEqual(x_pad.shape, tuple(shape))

    def test_correct_values_type1(self):
        """
        Test if the padding has the correct values.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.L = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)
        cp = grid.cp

        correct = cp.array([0, -2, -1, 0, 1, 2])

        for i in range(n_dims):
            p = Dirichlet(mset, i, btype=1)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            # check if the values are correct
            s = [0] * n_dims
            s[i] = slice(None)
            self.assertTrue(
                cp.allclose(x_pad[tuple(s)], correct)
            )

    def test_correct_values_type2(self):
        """
        Test if the padding has the correct values.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.L = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)
        cp = grid.cp

        correct = cp.array([-2, -1, 0, 0, 1, 2])

        for i in range(n_dims):
            p = Dirichlet(mset, i, btype=2)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            # check if the values are correct
            s = [0] * n_dims
            s[i] = slice(None)
            self.assertTrue(
                cp.allclose(x_pad[tuple(s)], correct)
            )


    def test_pad_unpad_type1(self):
        """
        Test if padding and unpadding are inverse operations.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)

        for i in range(n_dims):
            p = Dirichlet(mset, i, btype=1)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            x_unpad = p.unpad_from_fft(x_pad)
            # x and x_unpad should be the same
            self.assertTrue(np.allclose(x, x_unpad))

    def test_pad_unpad_type2(self):
        """
        Test if padding and unpadding are inverse operations.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)

        for i in range(n_dims):
            p = Dirichlet(mset, i, btype=2)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            x_unpad = p.unpad_from_fft(x_pad)
            # x and x_unpad should be the same
            self.assertTrue(np.allclose(x, x_unpad))

    
class TestNeumann(unittest.TestCase):
    """
    Test the Neumann boundary conditions.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """

        m = ModelSettingsBase(2)
        for i in range(2):
            b = Neumann(m, i, 1)
            self.assertEqual(b.mset, m)
            self.assertEqual(b.axis, i)

    def test_pad_shape_type1(self):
        """
        Test if the padding has the correct shape.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)

        for i in range(n_dims):
            p = Neumann(mset, i, btype=1)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            shape = list(x.shape)
            shape[i] *= 2
            self.assertEqual(x_pad.shape, tuple(shape))

    def test_correct_values_type1(self):
        """
        Test if the padding has the correct values.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.L = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)
        cp = grid.cp

        correct = cp.array([0, 2, 1, 0, 1, 2])

        for i in range(n_dims):
            p = Neumann(mset, i, btype=1)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            # check if the values are correct
            s = [0] * n_dims
            s[i] = slice(None)
            self.assertTrue(
                cp.allclose(x_pad[tuple(s)], correct)
            )

    def test_correct_values_type2(self):
        """
        Test if the padding has the correct values.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.L = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)
        cp = grid.cp

        correct = cp.array([2, 1, 0, 0, 1, 2])

        for i in range(n_dims):
            p = Neumann(mset, i, btype=2)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            # check if the values are correct
            s = [0] * n_dims
            s[i] = slice(None)
            self.assertTrue(
                cp.allclose(x_pad[tuple(s)], correct)
            )

    
    def test_pad_unpad_type1(self):
        """
        Test if padding and unpadding are inverse operations.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)

        for i in range(n_dims):
            p = Neumann(mset, i, btype=1)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            x_unpad = p.unpad_from_fft(x_pad)
            # x and x_unpad should be the same
            self.assertTrue(np.allclose(x, x_unpad))

    def test_pad_unpad_type2(self):
        """
        Test if padding and unpadding are inverse operations.
        """
        n_dims = 2
        mset = ModelSettingsBase(n_dims)
        mset.N = [3] * n_dims
        mset.periodic_bounds = [False, False]

        grid = GridBase(mset)

        for i in range(n_dims):
            p = Neumann(mset, i, btype=2)
            x = grid.X[i]
            x_pad = p.pad_for_fft(x)
            x_unpad = p.unpad_from_fft(x_pad)
            # x and x_unpad should be the same
            self.assertTrue(np.allclose(x, x_unpad))

    
class TestBoundaryConditions(unittest.TestCase):
    """
    Test the boundary conditions container.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """
        n_dims = 2

        m = ModelSettingsBase(n_dims)
        bounds = [Periodic(m, i) for i in range(n_dims)]
        b = BoundaryConditions(bounds)
        self.assertEqual(b.bounds, bounds)

    def test_shapes(self):
        """
        Test if shapes of the padded arrays are correct.
        """
        n_dim = 2 
        m = ModelSettingsBase(n_dim)
        m.N = [3] * n_dim
        bound_types = [Periodic, Dirichlet, Neumann]

        for btype in [1, 2]:
            for x_bound in bound_types:
                for y_bound in bound_types:
                    # set periodic setting
                    periods = [False] * n_dim
                    if x_bound == Periodic:
                        periods[0] = True
                    if y_bound == Periodic:
                        periods[1] = True
                    m.periodic_bounds = periods

                    grid = GridBase(m)
                    bounds = [x_bound(m, 0, btype), y_bound(m, 1, btype)]
                    b = BoundaryConditions(bounds)

                    x = grid.X[0]
                    x_pad = b.pad_for_fft(x)
                    shape = list(x.shape)
                    if x_bound != Periodic:
                        shape[0] *= 2
                    if y_bound != Periodic:
                        shape[1] *= 2
                    self.assertEqual(x_pad.shape, tuple(shape))

    def test_pad_unpad(self):
        """
        Test if padding and unpadding are inverse operations.
        """
        n_dim = 2 
        m = ModelSettingsBase(n_dim)
        m.N = [3] * n_dim
        bound_types = [Periodic, Dirichlet, Neumann]

        for btype in [1, 2]:
            for x_bound in bound_types:
                for y_bound in bound_types:
                    # set periodic setting
                    periods = [False] * n_dim
                    if x_bound == Periodic:
                        periods[0] = True
                    if y_bound == Periodic:
                        periods[1] = True
                    m.periodic_bounds = periods

                    grid = GridBase(m)
                    bounds = [x_bound(m, 0, btype), y_bound(m, 1, btype)]
                    b = BoundaryConditions(bounds)

                    x = grid.X[0]
                    x_pad = b.pad_for_fft(x)
                    x_unpad = b.unpad_from_fft(x_pad)

                    # x and x_unpad should be the same
                    self.assertTrue(grid.cp.allclose(x, x_unpad))

    def test_force_periodic(self):
        """
        Test if padding is periodic if forced.
        """
        n_dim = 2 
        m = ModelSettingsBase(n_dim)
        m.N = [3] * n_dim
        m.periodic_bounds = [True] * n_dim
        bound_types = [Periodic, Dirichlet, Neumann]

        for btype in [1, 2]:
            for x_bound in bound_types:
                for y_bound in bound_types:
                    grid = GridBase(m)
                    bounds = [x_bound(m, 0, btype), y_bound(m, 1, btype)]
                    b = BoundaryConditions(bounds)

                    x = grid.X[0]
                    x_pad = b.pad_for_fft(x)

                    # x and x_pad should be the same
                    self.assertTrue(grid.cp.allclose(x, x_pad))

                    x_unpad = b.unpad_from_fft(x_pad)
                    # x and x_unpad should be the same
                    self.assertTrue(grid.cp.allclose(x, x_unpad))
