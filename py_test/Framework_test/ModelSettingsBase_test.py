import unittest
import numpy as np

import sys
sys.path.append('../..')

from FIDOM.Framework.ModelSettingsBase import ModelSettingsBase

class TestModelSettingsBase(unittest.TestCase):
    """
    Test the base class for model settings container.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """
        # check if gpu is available
        try: 
            import cupy
            gpu_available = True
        except ImportError: 
            gpu_available = False

        for n_dims in range(1, 4):
            m = ModelSettingsBase(n_dims)
            self.assertEqual(m.n_dims, n_dims)
            self.assertEqual(m.dtype, np.float64)
            self.assertEqual(m.ctype, np.complex128)
            self.assertEqual(m.model_name, "Unnamed model")
            self.assertEqual(m.gpu, gpu_available)

    def test_set_L(self):
        """
        Test the setter for domain size.
        """

        n_dims = [2, 3]
        L = [[1.0, 2.0], [1.0, 2.0, 3.0]]
        N = [[64, 64], [64, 64, 64]]
        dg = [[1/64, 2/64], [1/64, 2/64, 3/64]]

        for n_dim, L_, N_, dg_ in zip(n_dims, L, N, dg):
            m = ModelSettingsBase(n_dim)
            m.L = L_
            self.assertEqual(m.L, L_)
            self.assertEqual(m.N, N_)
            self.assertEqual(m.dg, dg_)

    def test_set_N(self):
        """
        Test the setter for grid size.
        """

        n_dims = [2, 3]
        L = [[1.0, 1.0], [1.0, 1.0, 1.0]]
        N = [[64, 128], [32, 64, 128]]
        n_tot = [64*128, 32*64*128]
        dg = [[1/64, 1/128], [1/32, 1/64, 1/128]]

        for n_dim, L_, N_, dg_, n_tot_ in zip(n_dims, L, N, dg, n_tot):
            m = ModelSettingsBase(n_dim)
            m.N = N_
            self.assertEqual(m.L, L_)
            self.assertEqual(m.N, N_)
            self.assertEqual(m.dg, dg_)
            self.assertEqual(m.total_grid_points, n_tot_)
