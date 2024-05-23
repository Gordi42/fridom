import unittest
import numpy
import cupy
import os, sys
sys.path.append("../../../")

from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.Framework.FieldVariable import FieldVariable
from fridom.NonHydrostatic.Modules.Interpolation \
    .LinearInterpolation import LinearInterpolation

class LinearInterpolationTest(unittest.TestCase):
    def setUp(self):
        mset = ModelSettings(
            N = [3, 3, 3],
            periodic_bounds = [True, True, True]
        )
        grid = Grid(mset)
        self.inter_per = LinearInterpolation(mset, grid)
        mset= ModelSettings(
            N = [3, 3, 3],
            periodic_bounds = [False, False, False]
        )
        grid = Grid(mset)
        self.inter_nonper = LinearInterpolation(mset, grid)
        self.field = FieldVariable(mset, grid)
        self.field[:, 0, 0] = cupy.array([1, 2, 3])
        self.field[0, :, 0] = cupy.array([1, 2, 3])
        self.field[0, 0, :] = cupy.array([1, 2, 3])


    def test_sym_xf(self):
        # periodic
        xf = self.inter_per.sym_xf(self.field)
        # f[:,0,0] = [1, 2, 3] => xf[:,0,0] = [1.5, 2.5, 2]
        self.assertTrue(cupy.allclose(xf[:, 0, 0], cupy.array([1.5, 2.5, 2])))
        # non-periodic
        xf = self.inter_nonper.sym_xf(self.field)
        # f[:,0,0] = [1, 2, 3] => xf[:,0,0] = [1.5, 2.5, 1.5]
        self.assertTrue(cupy.allclose(xf[:, 0, 0], cupy.array([1.5, 2.5, 1.5])))

    def test_sym_xb(self):
        # periodic
        xb = self.inter_per.sym_xb(self.field)
        # f[:,0,0] = [1, 2, 3] => xb[:,0,0] = [2, 1.5, 2.5]
        self.assertTrue(cupy.allclose(xb[:, 0, 0], cupy.array([2, 1.5, 2.5])))
        # non-periodic
        xb = self.inter_nonper.sym_xb(self.field)
        # f[:,0,0] = [1, 2, 3] => xb[:,0,0] = [0.5, 1.5, 2.5]
        self.assertTrue(cupy.allclose(xb[:, 0, 0], cupy.array([0.5, 1.5, 2.5])))

    def test_sym_yf(self):
        # periodic
        yf = self.inter_per.sym_yf(self.field)
        # f[0,:,0] = [1, 2, 3] => yf[0,:,0] = [1.5, 2.5, 2]
        self.assertTrue(cupy.allclose(yf[0, :, 0], cupy.array([1.5, 2.5, 2])))
        # non-periodic
        yf = self.inter_nonper.sym_yf(self.field)
        # f[0,:,0] = [1, 2, 3] => yf[0,:,0] = [1.5, 2.5, 1.5]
        self.assertTrue(cupy.allclose(yf[0, :, 0], cupy.array([1.5, 2.5, 1.5])))

    def test_sym_yb(self):
        # periodic
        yb = self.inter_per.sym_yb(self.field)
        # f[0,:,0] = [1, 2, 3] => yb[0,:,0] = [2, 1.5, 2.5]
        self.assertTrue(cupy.allclose(yb[0, :, 0], cupy.array([2, 1.5, 2.5])))
        # non-periodic
        yb = self.inter_nonper.sym_yb(self.field)
        # f[0,:,0] = [1, 2, 3] => yb[0,:,0] = [0.5, 1.5, 2.5]
        self.assertTrue(cupy.allclose(yb[0, :, 0], cupy.array([0.5, 1.5, 2.5])))

    def test_sym_zf(self):
        # periodic
        zf = self.inter_per.sym_zf(self.field)
        # f[0,0,:] = [1, 2, 3] => zf[0,0,:] = [1.5, 2.5, 2]
        self.assertTrue(cupy.allclose(zf[0, 0, :], cupy.array([1.5, 2.5, 2])))
        # non-periodic
        zf = self.inter_nonper.sym_zf(self.field)
        # f[0,0,:] = [1, 2, 3] => zf[0,0,:] = [1.5, 2.5, 1.5]
        self.assertTrue(cupy.allclose(zf[0, 0, :], cupy.array([1.5, 2.5, 1.5])))

    def test_sym_zb(self):
        # periodic
        zb = self.inter_per.sym_zb(self.field)
        # f[0,0,:] = [1, 2, 3] => zb[0,0,:] = [2, 1.5, 2.5]
        self.assertTrue(cupy.allclose(zb[0, 0, :], cupy.array([2, 1.5, 2.5])))
        # non-periodic
        zb = self.inter_nonper.sym_zb(self.field)
        # f[0,0,:] = [1, 2, 3] => zb[0,0,:] = [0.5, 1.5, 2.5]
        self.assertTrue(cupy.allclose(zb[0, 0, :], cupy.array([0.5, 1.5, 2.5])))