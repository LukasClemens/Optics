import numpy as np
import cmath
from abstraction import Stack, Material, Slab

class Spectra:
    def __init__(self, stack: Stack, start: float, stop: float, step: float):
        self.stack = stack
        self.start = start
        self.stop = stop
        self.step = step

    def make_array(self):
        wavelengths = np.arange(self.start, self.stop, self.step)
        transmission_values = []
        reflectance_values = []
        for wavelength in wavelengths:
            matrix = self.stack.make_matrix(wavelength)

            # entries of the transfer matrix
            a = matrix[0, 0]
            b = matrix[0, 1]
            c = matrix[1, 0]
            d = matrix[1, 1]

            # field reflectance and transmission
            t = (a * d - b * c) / d
            r = -c / d


            n_medium = self.stack.medium.get_refr_ind(wavelength)
            n_substrate = self.stack.substrate.get_refr_ind(wavelength)
            reflectance = abs(r) ** 2
            transmission = abs(t) ** 2 * n_substrate / n_medium

            transmission_values.append(transmission)
            reflectance_values.append(reflectance)

        return wavelengths, np.array(transmission_values), np.array(reflectance_values)