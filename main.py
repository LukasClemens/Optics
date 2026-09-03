import numpy as np

from calc import Spectra
from abstraction import Stack, material_dict, Material

if __name__ == '__main__':
    medium = Material(*material_dict[0])
    substrate = Material(*material_dict[4])
    stack = Stack(medium=medium,
                  substrate=substrate,
                  stack_file='designs/bragg_20.txt',
                  target_wavelength=700)
    spectra = Spectra(stack=stack, start=400, stop=1000, step=1)
    wavelengths, transmission, reflectance = spectra.make_array()
    np.savetxt(fname='results/bragg_20.txt',
               X=np.vstack((wavelengths, transmission, reflectance)).T,
               header=f'lambda\tT\tR')