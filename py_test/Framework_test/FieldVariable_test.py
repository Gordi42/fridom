import unittest
import numpy
import os, sys
sys.path.append("../..")

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase
from fridom.Framework.BoundaryConditions import *
from fridom.Framework.FieldVariable import FieldVariable

class TestFieldVariable(unittest.TestCase):
    """
    Test the FieldVariable class
    """
    # ==================================================================
    #  TEST CONSTRUCTORS
    # ==================================================================
    def get_gpu_list(self):
        """
        Return a list of booleans weather to test on gpu or not
        """
        gpu_list = [False]
        try:
            import cupy
            gpu_list.append(True)
        except ImportError:
            pass
        return gpu_list

    def test_constructor(self):
        """
        Test the Constructor with a given input array
        """
        gpu_list = self.get_gpu_list()
        n_dims = [1,2,3]
        spectrals = [True, False]

        for gpu in gpu_list:
            for n_dim in n_dims:
                for spectral in spectrals:
                    m = ModelSettingsBase(n_dim)
                    m.gpu = gpu
                    m.N = [1] * n_dim
                    g = GridBase(m)
                    cp = g.cp
                    arr = cp.ones(shape=tuple(m.N))
                    field = FieldVariable(
                        m, g, is_spectral=spectral, name="Test", arr=arr)
                    
                    dtype = m.ctype if spectral else m.dtype
                    self.assertEqual(field.arr.dtype, dtype)
                    self.assertEqual(cp.allclose(field.arr, arr), True)
                    self.assertEqual(field.name, "Test")

    def test_zeros(self):
        """
        Test the construction with zeros
        """
        gpu_list = self.get_gpu_list()
        n_dims = [1,2,3]
        spectrals = [True, False]

        for gpu in gpu_list:
            for n_dim in n_dims:
                for spectral in spectrals:
                    m = ModelSettingsBase(n_dim)
                    m.gpu = gpu
                    m.N = [1] * n_dim
                    g = GridBase(m)
                    cp = g.cp
                    arr = cp.zeros(shape=tuple(m.N))
                    field = FieldVariable(
                        m, g, is_spectral=spectral, name="Test")
                    
                    dtype = m.ctype if spectral else m.dtype
                    self.assertEqual(field.arr.dtype, dtype)
                    self.assertEqual(cp.allclose(field.arr, arr), True)
                    self.assertEqual(field.name, "Test")

    def test_copy(self):
        """
        Test the copy method
        """
        gpu_list = self.get_gpu_list()
        n_dims = [1,2,3]
        spectrals = [True, False]

        for gpu in gpu_list:
            for n_dim in n_dims:
                for spectral in spectrals:
                    m = ModelSettingsBase(n_dim)
                    m.gpu = gpu
                    m.N = [1] * n_dim
                    g = GridBase(m)
                    cp = g.cp
                    field = FieldVariable(
                        m, g, is_spectral=spectral, name="Test")

                    copy = field.copy()
                    # Test that the copy is not the same object
                    self.assertFalse(copy is field)
                    # Test that the copy has the same values
                    self.assertTrue(cp.all(copy.arr == field.arr))
                    # Change the copy and test that the original is not changed
                    copy.arr[:] = 1
                    self.assertFalse(cp.all(copy.arr == field.arr))
                    

    # ==================================================================
    #  TEST METHODS
    # ==================================================================
    def test_fft_ifft(self):
        """
        Test whether fft called two times is identity.
        """
        gpu_list = self.get_gpu_list()
        n_dims = [1,2,3]

        bc = [Periodic, Dirichlet, Neumann]
        periodic = [True, False, False]

        for gpu in gpu_list:
            for n_dim in n_dims:
                for bc_, periodic_ in zip(bc, periodic):
                    m = ModelSettingsBase(n_dim)
                    m.gpu = gpu
                    m.periodic_bounds = [periodic_] * n_dim
                    # create array of random integers between 1 and 10
                    m.N = [numpy.random.randint(1,10) for i in range(n_dim)]
                    bounds = [bc_(m, i, 1) for i in range(n_dim)]
                    boundary_conditions = BoundaryConditions(bounds)
                    g = GridBase(m)
                    cp = g.cp
                    field = FieldVariable(
                        m, g, is_spectral=False, name="Test", bc=boundary_conditions)
                    field.arr = cp.random.rand(*m.N)

                    field_hat = field.fft()
                    field_hat_hat = field_hat.fft()

                    # Test that the fft is the inverse of itself
                    self.assertTrue(cp.allclose(field.arr, field_hat_hat.arr))



    def test_fft(self):
        """
        Test the fft method
        """
        gpu_list = self.get_gpu_list()
        n_dim = 3

        bcs = [Periodic, Dirichlet, Neumann]
        periods = [True, False, False]
        aims = [6, 0, 6*8]

        for gpu in gpu_list:
            for bc, period, aim in zip(bcs, periods, aims):
                m = ModelSettingsBase(n_dim)
                m.gpu = gpu
                m.periodic_bounds = [period] * n_dim
                m.N = [3, 2, 1]
                bounds = [bc(m, i, 1) for i in range(n_dim)]
                boundary_conditions = BoundaryConditions(bounds)
                g = GridBase(m)
                field = FieldVariable(
                    m, g, is_spectral=False, name="Test", bc=boundary_conditions)
                field.arr[:] = 1

                fft = field.fft()
                # zero entry should be sum
                self.assertEqual(fft.arr[0,0,0], aim)

    def test_sqrt(self):
        """
        Test the sqrt method
        """
        gpu_list = self.get_gpu_list()
        n_dims = [1,2,3]
        spectrals = [False]

        for gpu in gpu_list:
            for n_dim in n_dims:
                for spectral in spectrals:
                    m = ModelSettingsBase(n_dim)
                    m.gpu = gpu
                    m.N = [1] * n_dim
                    g = GridBase(m)
                    cp = g.cp
                    field = FieldVariable(
                        m, g, is_spectral=spectral, name="Test")
                    field.arr[:] = 2

                    sqrt = field.sqrt()
                    self.assertTrue(cp.allclose(sqrt.arr, 1.4142135623730951))

    def test_norm_l2(self):
        """
        Test the norm_l2 method
        """
        gpu_list = self.get_gpu_list()
        n_dims = [1,2,3]
        spectrals = [False]

        for gpu in gpu_list:
            for n_dim in n_dims:
                for spectral in spectrals:
                    m = ModelSettingsBase(n_dim)
                    m.gpu = gpu
                    m.N = [2] * n_dim
                    g = GridBase(m)
                    cp = g.cp
                    field = FieldVariable(
                        m, g, is_spectral=spectral, name="Test")
                    field.arr[:] = 2

                    norm = field.norm_l2()
                    self.assertEqual(norm, cp.linalg.norm(field.arr))

    def test_pad_raw(self):
        """
        Test the pad_raw method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [1, 3, 1]
            g = GridBase(m)
            cp = g.cp
            field = FieldVariable(
                m, g, is_spectral=False, name="Test")
            field.arr[0,:,0] = cp.array([1,2,3])

            pad = field.pad_raw(pad_width=((0,0),(0,0),(0,0)))
            self.assertTrue(cp.allclose(pad, field.arr))

            pad = field.pad_raw(pad_width=((0,0),(1,1),(0,0)))
            aim = [3,1,2,3,1]
            self.assertTrue(cp.allclose(pad[0,:,0], aim))

    def test_ave(self):
        """
        Test the ave method
        """
        gpu_list = self.get_gpu_list()
        periods = [True, True, False, False]
        shifts = [1, -1, 1, -1]
        aims = [
            [1.5, 2.5, 2],
            [2, 1.5, 2.5],
            [1.5, 2.5, 1.5],
            [0.5, 1.5, 2.5]
        ]

        for gpu in gpu_list:
            for period, shift, aim in zip(periods, shifts, aims):
                m = ModelSettingsBase(3)
                m.gpu = gpu
                m.N = [1, 3, 1]
                m.periodic_bounds = [period] * 3
                g = GridBase(m)
                cp = g.cp
                field = FieldVariable(
                    m, g, is_spectral=False, name="Test")
                field.arr[0,:,0] = cp.array([1,2,3])

                ave = field.ave(axis=1, shift=shift)
                self.assertTrue(cp.allclose(ave.arr[0,:,0], aim))

    def test_diff_forward(self):
        """
        Test the diff_forward method
        """
        gpu_list = self.get_gpu_list()
        periods = [True, False]
        aims = [
            [1*3, 1*3, -2*3],
            [1*3, 1*3, -3*3],
        ]

        for gpu in gpu_list:
            for period, aim in zip(periods, aims):
                m = ModelSettingsBase(3)
                m.gpu = gpu
                m.N = [1, 3, 1]
                self.assertAlmostEqual(m.dg[1], 1/3)
                m.periodic_bounds = [period] * 3
                g = GridBase(m)
                cp = g.cp
                field = FieldVariable(
                    m, g, is_spectral=False, name="Test")
                field.arr[0,:,0] = cp.array([1,2,3])

                diff = field.diff_forward(axis=1)
                self.assertTrue(cp.allclose(diff.arr[0,:,0], aim))

    def test_diff_backward(self):
        """
        Test the diff_backward method
        """
        gpu_list = self.get_gpu_list()
        periods = [True, False]
        aims = [
            [-2*3, 1*3, 1*3],
            [1*3, 1*3, 1*3],
        ]

        for gpu in gpu_list:
            for period, aim in zip(periods, aims):
                m = ModelSettingsBase(3)
                m.gpu = gpu
                m.N = [1, 3, 1]
                self.assertAlmostEqual(m.dg[1], 1/3)
                m.periodic_bounds = [period] * 3
                g = GridBase(m)
                cp = g.cp
                field = FieldVariable(
                    m, g, is_spectral=False, name="Test")
                field.arr[0,:,0] = cp.array([1,2,3])

                diff = field.diff_backward(axis=1)
                self.assertTrue(cp.allclose(diff.arr[0,:,0], aim))


    # ==================================================================
    #  ARRAY INDEXING
    # ==================================================================

    def test_getitem(self):
        """
        Test the __getitem__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            g = GridBase(m)
            cp = g.cp
            zeros = FieldVariable(
                m, g, is_spectral=False, name="Test")

            # Test single index
            value = zeros[0,0,0]
            self.assertEqual(value, 0)
            # Test slice
            values = zeros[:,0,0]
            self.assertEqual(cp.all(values == cp.zeros(3)), True)


    def test_setitem(self):
        """
        Test the __setitem__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            g = GridBase(m)
            cp = g.cp
            zeros = FieldVariable(
                m, g, is_spectral=False, name="Test")

            # Test single index
            zeros[0,0,0] = 1
            self.assertEqual(zeros.arr[0,0,0], 1)

            # Test slice
            zeros[0,0,:] = 2
            self.assertEqual(zeros.arr[0,0,-1], 2)


    def test_getattr(self):
        """
        Test the __getattr__ method (used to apply numpy methods to the array)
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            g = GridBase(m)
            cp = g.cp
            ones = FieldVariable(
                m, g, is_spectral=False, name="Test")
            ones[:] = 1

            # test sum
            sum = cp.sum(ones)
            self.assertEqual(sum, 6)

            # test shape
            shape = ones.shape
            self.assertEqual(shape, (3,2,1))

            # test cosine
            cos = cp.cos(ones)
            self.assertEqual(cos[0,0,0], cp.cos(1))
            

    # ==================================================================
    #  ARITHMETIC OPERATORS
    # ==================================================================
    def test_add(self):
        """
        Test the __add__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            zeros = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test sum with scalar
            sum = zeros + 1
            self.assertEqual(sum.arr[0,0,0], 1)

            # Test sum with FieldVariable
            sum = zeros + ones
            self.assertEqual(sum.arr[0,0,0], 1)

            # Test sum with array
            sum = zeros + zeros.cp.ones(grid.X[0].shape)
            self.assertEqual(sum.arr[0,0,0], 1)

    def test_radd(self):
        """
        Test the __radd__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            zeros = FieldVariable(
                m, grid, is_spectral=False, name="Test")

            # Test sum with scalar
            sum = 1 + zeros
            self.assertEqual(sum.arr[0,0,0], 1)

            # Test sum with array
            ones = cp.ones(grid.X[0].shape)
            sum = ones + zeros
            self.assertEqual(sum[0,0,0], 1)

    def test_sub(self):
        """
        Test the __sub__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            zeros = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test subtraction with scalar
            sub = zeros - 1
            self.assertEqual(sub.arr[0,0,0], -1)

            # Test subtraction with FieldVariable
            sub = zeros - ones
            self.assertEqual(sub.arr[0,0,0], -1)

            # Test subtraction with array
            sub = zeros - cp.ones(grid.X[0].shape)
            self.assertEqual(sub.arr[0,0,0], -1)

    def test_rsub(self):
        """
        Test the __rsub__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            zeros = FieldVariable(
                m, grid, is_spectral=False, name="Test")

            # Test subtraction with scalar
            sub = 1 - zeros
            self.assertEqual(sub.arr[0,0,0], 1)

            # Test subtraction with array
            ones = cp.ones(grid.X[0].shape)
            sub = ones - zeros
            self.assertEqual(sub[0,0,0], 1)

    def test_mul(self):
        """
        Test the __mul__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            zeros = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test multiplication with scalar
            mul = ones * 2
            self.assertEqual(mul.arr[0,0,0], 2)

            # Test multiplication with FieldVariable
            mul = zeros * ones
            self.assertEqual(mul.arr[0,0,0], 0)

            # Test multiplication with array
            mul = ones * cp.ones(grid.X[0].shape)
            self.assertEqual(mul.arr[0,0,0], 1)

    def test_rmul(self):
        """
        Test the __rmul__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            zeros = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test multiplication with scalar
            mul = 2 * ones
            self.assertEqual(mul.arr[0,0,0], 2)

            # Test multiplication with array
            mul = cp.ones(grid.X[0].shape) * zeros
            self.assertEqual(mul[0,0,0], 0)

    def test_truediv(self):
        """
        Test the __truediv__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test division with scalar
            div = ones / 2
            self.assertEqual(div.arr[0,0,0], 0.5)

            # Test division with FieldVariable
            div = ones / ones
            self.assertEqual(div.arr[0,0,0], 1)

            # Test division with array
            div = ones / (cp.ones(grid.X[0].shape)*2)
            self.assertEqual(div.arr[0,0,0], 0.5)

    def test_rtruediv(self):
        """
        Test the __rtruediv__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test division with scalar
            div = 2 / ones
            self.assertEqual(div.arr[0,0,0], 2)

            # Test division with array
            div = (cp.ones(grid.X[0].shape)*2) / ones
            self.assertEqual(div[0,0,0], 2)

    def test_pow(self):
        """
        Test the __pow__ method
        """
        gpu_list = self.get_gpu_list()

        for gpu in gpu_list:
            m = ModelSettingsBase(3)
            m.gpu = gpu
            m.N = [3, 2, 1]
            grid = GridBase(m)
            cp = grid.cp
            ones = FieldVariable(
                m, grid, is_spectral=False, name="Test")
            ones[:] = 1

            # Test power with scalar
            pow = ones ** 2
            self.assertEqual(pow.arr[0,0,0], 1)

            # Test power with FieldVariable
            pow = ones ** ones
            self.assertEqual(pow.arr[0,0,0], 1)

            # Test power with array
            pow = ones ** (cp.ones(grid.X[0].shape))
            self.assertEqual(pow.arr[0,0,0], 1)
    