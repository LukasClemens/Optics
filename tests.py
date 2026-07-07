import unittest
from abstraction import Stack, Material, material_dict
import numpy


class TestStack(unittest.TestCase):
    def test_init_non_dispersive(self):
        medium = Material(*material_dict[0])
        substrate = Material(*material_dict[4])
        stack = Stack(medium=medium, substrate=substrate, stack_file='designs/test_no_dispersion.txt')
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
        self.assertIs(type(stack.slab_stack[1].material.refr_ind), numpy.ndarray)
        self.assertListEqual(list(stack.slab_stack[1].material.refr_ind[0]), [1, 2, 3])
        # makes sure the thickness of a layer is a real number
        self.assertFalse(stack.slab_stack[1].thickness.imag)


if __name__ == '__main__':
    unittest.main()
