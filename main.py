import numpy as np

from calc import Calc
from abstraction import Stack, material_dict, Material
from visualisation import Visualiser

if __name__ == '__main__':
    medium = Material(*material_dict[0])
    substrate = Material(*material_dict[4])
    stack = Stack(medium=medium,
                  substrate=substrate,
                  stack_file='designs/cavity.txt',
                  target_wavelength=700)

    # enable the following line for the silver cavity
    # stack.slab_stack[10].thickness = 5

    # enable to set thickness for AR coating
    # stack.slab_stack[0].thickness = 80.69
    calc = Calc(stack=stack)

    # enable to compare coefficients
    # print('R and T obtained from matrix')
    # print('R: {}\tT: {}'.format(*calc.get_r_t_from_matrix(700)))
    # print('R and T obtained from series:')
    # print('R: {}\tT: {}'.format(*calc.get_r_t_from_series(700)))
    visualiser = Visualiser(calc=calc)
    visualiser.animate_efield(1)