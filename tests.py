import unittest
from abstraction import Stack, Material, material_dict, PropMatrix, InterMatrix
import numpy as np


class AbstractionTest(unittest.TestCase):
    def test_init_non_dispersive(self):
        medium = Material(*material_dict[0])
        substrate = Material(*material_dict[4])
        stack = Stack(medium=medium,
                      substrate=substrate,
                      stack_file='designs/test_no_dispersion.txt',
                      target_wavelength=700)
        self.assertEqual(len(stack.slab_stack), 4)
        self.assertEqual(stack.slab_stack[1].material.refr_ind, 1.6)
        self.assertTrue(stack.slab_stack[1].thickness)

    def test_init_dispersive(self):
        medium = Material(*material_dict[0])
        substrate = Material(*material_dict[4])
        stack = Stack(medium=medium,
                      substrate=substrate,
                      stack_file='designs/test_dispersion.txt',
                      target_wavelength=2.5)
        self.assertIs(type(stack.slab_stack[1].material.refr_ind), np.ndarray)
        self.assertListEqual(list(stack.slab_stack[1].material.refr_ind[0]), [1, 2, 3])
        # makes sure the thickness of a layer is a real number
        self.assertFalse(stack.slab_stack[1].thickness.imag)

    def test_propmatrix(self):
        prop_matrix = PropMatrix(100, 700, Material(*material_dict[4]))
        refractive_index = prop_matrix.refr_ind
        k_number = prop_matrix.k_number
        matrix = prop_matrix.matrix
        solution_matrix = np.array([[complex(0.26605, -0.96396), 0], [0, complex(0.26605, 0.96396)]])
        self.assertAlmostEqual(refractive_index, 1.45)
        self.assertAlmostEqual(k_number, 0.013015, places=5)
        for element in list(zip(matrix.flatten(), solution_matrix.flatten())):
            self.assertAlmostEqual(element[0], element[1], places=4)

    def test_intermatrix(self):
        inter_matrix = InterMatrix(Material(*material_dict[4]), Material(*material_dict[2]), 700)
        matrix = inter_matrix.matrix
        solution_matrix = np.array([[0.82955, 0.17045], [0.17045, 0.82955]])
        for element in list(zip(matrix.flatten(), solution_matrix.flatten())):
            self.assertAlmostEqual(element[0], element[1], places=4)

    def test_stack(self):
        medium = Material(*material_dict[0])
        substrate = Material(*material_dict[4])
        stack = Stack(medium=medium,
                      substrate=substrate,
                      stack_file='designs/test_no_dispersion.txt',
                      target_wavelength=700)
        matrix = stack.make_matrix(700)
        solution = np.array([1.41470, 1.14530, 1.14530, 1.41470])
        for element in list(zip(matrix.flatten(), solution)):
            self.assertAlmostEqual(element[0], element[1], places=4)


if __name__ == '__main__':
    unittest.main()
