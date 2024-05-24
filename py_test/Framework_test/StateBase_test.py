import unittest
import numpy
import cupy
import os, sys
sys.path.append("../..")

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase
from fridom.Framework.BoundaryConditions import *
from fridom.Framework.FieldVariable import FieldVariable
from fridom.Framework.StateBase import StateBase

class StateBaseTest(unittest.TestCase):
    def setUp(self):
        n_dims = 2
        self.mset = mset = ModelSettingsBase(n_dims)
        self.mset.N = [10, 10]
        bounds = [Periodic(mset, 0), Periodic(mset, 1)]
        self.grid = grid = GridBase(mset)
        bc = BoundaryConditions(bounds)
        self.fields = [FieldVariable(grid, is_spectral=False, bc=bc) 
                       for i in range(3)]
        
        self.mset_1d = mset_1d = ModelSettingsBase(1)
        self.mset_1d.N = [3]
        bounds = [Periodic(mset_1d, 0)]
        self.grid_1d = grid_1d = GridBase(mset_1d)
        bc = BoundaryConditions(bounds)
        self.zeros_p = FieldVariable(grid_1d,
                                       is_spectral=False, bc=bc)
        self.ones_p = self.zeros_p + 1
        self.zeros_s = FieldVariable(grid_1d,
                                        is_spectral=True, bc=bc)
        self.ones_s = self.zeros_s + 1
        self.imag_s = self.zeros_s + 1j


    def test_init(self):
        state = StateBase(self.grid, self.fields)
        self.assertEqual(state.mset, self.mset)
        self.assertEqual(state.grid, self.grid)
        self.assertEqual(state.field_list, self.fields)

    def test_copy(self):
        state = StateBase(self.grid, self.fields)
        state2 = state.copy()
        self.assertEqual(state2.mset, self.mset)
        self.assertEqual(state2.grid, self.grid)
        self.assertNotEqual(state2.field_list, state.field_list)

        # change state2
        state2.field_list[0][0,0] = 1

        # check that state1 is unchanged
        self.assertEqual(state.field_list[0][0,0], 0)

    def test_fft(self):
        state = StateBase(self.grid, self.fields)
        for field in state.field_list:
            field[:] = state.cp.random.rand(*field.shape)
        state_fft = state.fft()
        self.assertEqual(state_fft.mset, self.mset)
        self.assertEqual(state_fft.grid, self.grid)
        self.assertNotEqual(state_fft.field_list, state.field_list)
        self.assertEqual(state_fft.is_spectral, True)
        cp = state.cp

        # second fft
        state_fft_fft = state_fft.fft()

        # check if fft(fft) is identity
        fields_in = state.field_list
        fields_out = state_fft_fft.field_list
        for f1, f2 in zip(fields_in, fields_out):
            self.assertTrue(cp.allclose(f1, f2))

    def test_dot(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])
        cp = state.cp

        dot = state.dot(state2)
        self.assertTrue(isinstance(dot, FieldVariable))
        self.assertTrue(cp.allclose(dot[:], self.ones_p[:]))

        # test complex
        state = StateBase(self.grid_1d, [self.ones_s, self.ones_s - 1j])
        state2 = StateBase(self.grid_1d, [self.imag_s, self.ones_s])

        dot = state.dot(state2)
        self.assertTrue(cp.allclose(dot[:], self.ones_s[:]-2j))

    def test_norm_l2(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        # yields state = [(0, 1), (0, 1), (0, 1)]
        # with l2 norm = sqrt((1+1+1) * 1/3) = 1
        #                               ^^^
        #                               dV
        self.assertEqual(state.norm_l2(), 1)

        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])
        # yields state2 = [(1, 1), (1, 1), (1, 1)]
        # with l2 norm = sqrt((2+2+2) * 1/3) = sqrt(2)
        self.assertEqual(state2.norm_l2(), 2**(1/2))

    def test_norm_of_diff(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])

        # test norm of difference between two identical states
        # should be 0
        norm = state.norm_of_diff(state)
        self.assertEqual(norm, 0)

        # test norm of difference between two different states
        # the l2 norm of state - state2 is:
        # sqrt((1+1+1) * 1/3) = 1
        self.assertEqual((state - state2).norm_l2(), 1)

        # hence, the norm of the difference should be
        # 2 * |z - z'|_2 / (|z|_2 + |z'|_2)
        # 2 *    1       / ( 1    + 2**(1/2))
        norm = state.norm_of_diff(state2)
        self.assertEqual(norm, 2 / (1 + 2**(1/2)))

    def test_add(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])
        cp = state.cp

        # add two states
        state3 = state + state2
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

        # add state and field
        state3 = state + self.ones_p
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

        # add state and number
        state3 = state + 1
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

        # add state and array
        state3 = state + self.ones_p[:]
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

        # add number and state
        state3 = 1 + state
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

    def test_sub(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])
        cp = state.cp

        # sub two states
        state3 = state - state2
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], -self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 0*self.ones_p[:]))

        # sub state and field
        state3 = state - self.ones_p
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], -self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 0*self.ones_p[:]))

        # sub state and number
        state3 = state - 1
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], -self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 0*self.ones_p[:]))

        # sub state and array
        state3 = state - self.ones_p[:]
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], -self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 0*self.ones_p[:]))

        # sub number and state
        state3 = 1 - state
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 0*self.ones_p[:]))

    def test_mul(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])
        cp = state.cp

        # mul two states
        state3 = state * state2
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], self.ones_p[:]))

        # mul state and field
        state3 = state * self.ones_p
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], self.ones_p[:]))

        # mul state and number
        state3 = state * 2
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

        # mul state and array
        state3 = state * self.ones_p[:]
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], self.ones_p[:]))

        # mul number and state
        state3 = 2 * state
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))

    def test_truediv(self):
        state = StateBase(self.grid_1d, [self.zeros_p, self.ones_p])
        state2 = StateBase(self.grid_1d, [self.ones_p, self.ones_p])
        cp = state.cp

        # div two states
        state3 = state / state2
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], self.ones_p[:]))

        # div state and field
        state3 = state / self.ones_p
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], self.ones_p[:]))

        # div state and number
        state3 = state / 2
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 0.5*self.ones_p[:]))

        # div state and array
        state3 = state / self.ones_p[:]
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], self.zeros_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], self.ones_p[:]))

        # div number and state
        state3 = 2 / state
        self.assertTrue(isinstance(state3, StateBase))
        self.assertTrue(cp.allclose(state3.field_list[0][:], numpy.inf*self.ones_p[:]))
        self.assertTrue(cp.allclose(state3.field_list[1][:], 2*self.ones_p[:]))


    
        