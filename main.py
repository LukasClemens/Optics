import numpy as np

from calc import Calc
from abstraction import Stack, material_dict, Material

if __name__ == '__main__':
    medium = Material(*material_dict[0])
    substrate = Material(*material_dict[4])
    stack = Stack(medium=medium,
                  substrate=substrate,
                  stack_file='designs/bragg_20.txt',
                  target_wavelength=700)
    calc = Calc(stack=stack)
    position, field = calc.make_efield_array(1, 700)
    np.savetxt(fname='results/field_bragg_20.txt',
               X=np.vstack((position, field)).T,
               header=f'z\tE')