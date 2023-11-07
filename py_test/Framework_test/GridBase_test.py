import unittest
import numpy as np

import sys
sys.path.append('../..')

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase

class TestGridBase(unittest.TestCase):
    """
    Test the base class for grid container.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """

        for n_dims in range(1, 3):
            m = ModelSettingsBase(n_dims)
            g = GridBase(m)
            self.assertEqual(g.mset, m)

    def test_x(self):
        """
        Test the physical domain.
        """

        n_dims = [1, 2]
        L = [[1.0], [1.0, 2.0]]
        N = [[64], [32, 128]]
        dg = [[1/64], [1/32, 2/128]]
        print(dg)

        for n_dim, L_, N_, dg_ in zip(n_dims, L, N, dg):
            m = ModelSettingsBase(n_dim)
            m.L = L_
            m.N = N_
            g = GridBase(m)
            self.assertEqual(len(g.x), n_dim)
            for i in range(n_dim):
                self.assertEqual(g.x[i][0], 0)
                self.assertEqual(g.x[i][-1], L_[i] - dg_[i])

    def test_X(self):
        """
        Test the physical domain (meshgrid).
        """

        n_dims = [1, 2]
        L = [[1.0], [1.0, 2.0]]
        N = [[64], [32, 128]]
        dg = [[1/64], [1/32, 2/128]]

        for n_dim, L_, N_, dg_ in zip(n_dims, L, N, dg):
            m = ModelSettingsBase(n_dim)
            m.L = L_
            m.N = N_
            g = GridBase(m)
            self.assertEqual(len(g.X), n_dim)
            for i in range(n_dim):
                self.assertEqual(g.X[i].shape, tuple(N_))

    def test_k(self):
        """
        Test the spectral domain.
        """

        n_dims = [1, 2]
        L = [[1.0], [1.0, 2.0]]
        N = [[64], [32, 128]]
        dg = [[1/64], [1/32, 2/128]]
        k_max = [[np.pi*64], [np.pi*32, np.pi*64]]

        for n_dim, L_, N_, dg_, k_max_ in zip(n_dims, L, N, dg, k_max):
            m = ModelSettingsBase(n_dim)
            m.L = L_
            m.N = N_
            g = GridBase(m)
            cp = g.cp
            self.assertEqual(len(g.k), n_dim)
            for i in range(n_dim):
                self.assertEqual(g.k[i][0], 0)
                self.assertEqual(max(cp.abs(g.k[i])), k_max_[i])

    def test_K(self):
        """
        Test the spectral domain (meshgrid).
        """

        n_dims = [1, 2]
        L = [[1.0], [1.0, 2.0]]
        N = [[64], [32, 128]]
        dg = [[1/64], [1/32, 2/128]]
        k_max = [[np.pi*64], [np.pi*32, np.pi*64]]

        for n_dim, L_, N_, dg_, k_max_ in zip(n_dims, L, N, dg, k_max):
            m = ModelSettingsBase(n_dim)
            m.L = L_
            m.N = N_
            g = GridBase(m)
            self.assertEqual(len(g.K), n_dim)
            for i in range(n_dim):
                self.assertEqual(g.K[i].shape, tuple(N_))

    def test_padding(self):
        """
        Test the padding of the spectral domain.
        """

        n_dims = 2
        L = [1, 1]; N = [64, 64]; dg = [1/64, 1/64]
        periods = [True, False]

        m = ModelSettingsBase(n_dims)
        m.L = L
        m.N = N
        m.periodic_bounds = periods

        g = GridBase(m)

        self.assertEqual(g.X[0].shape, (64, 64))
        self.assertEqual(g.K[0].shape, (64, 128))

        # nyquist should be same for both
        cp = g.cp
        self.assertEqual(cp.max(cp.abs(g.k[0])), cp.max(cp.abs(g.k[1])))


