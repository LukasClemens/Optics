import numpy as np
import cmath

from numpy.ma.core import floor

from abstraction import Stack, Material, Slab, PropMatrix
from typing import List

class Calc:
    def __init__(self, stack: Stack):
        self.stack = stack

    def make_spectrum_array(self, start: float, stop: float, step: float):
        wavelengths = np.arange(start, stop, step)
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


    def make_efield_array(self, step_width: float, wavelength: float|None = None):
        position_list = []
        field_list = []
        if not wavelength:
            wavelength = self.stack.target_wavelength
        matrix = self.stack.make_matrix(wavelength)

        incident_amplitude = 0.5
        # entries of the transfer matrix
        c = matrix[1, 0]
        d = matrix[1, 1]
        reflected_amplitude = -incident_amplitude * c / d

        n_inc = self.stack.medium.get_refr_ind(wavelength)
        k_inc = 2 * np.pi *n_inc / wavelength

        # recreates E-field in incident medium
        for z in np.arange(0, -wavelength/2, -step_width):
            field = incident_amplitude * cmath.exp(1j * k_inc * z) + reflected_amplitude * cmath.exp(-1j * k_inc * z)
            position_list.append(z)
            field_list.append(abs(field.real))

        # acquires a list of all interface and propagation matrices in reversed order (closest to incidence first)
        reversed_matrix_list = self.stack.make_matrix_list(wavelength)
        interface_matrices = reversed_matrix_list[::2]
        slabs: List[Slab] = [slab for slab in self.stack.slab_stack]
        slabs.reverse()

        # initialises left-side field amplitudes
        positive_wave = incident_amplitude
        negative_wave = reflected_amplitude
        # initialises thickness to zero
        starting_thickness = 0
        for pair in zip(interface_matrices, slabs):
            interface_matrix = pair[0]
            slab = pair[1]
            material = slab.material
            slab_thickness = slab.thickness

            # calculates amplitudes on the right side of the interface
            positive_wave, negative_wave = tuple(np.inner(interface_matrix, np.array([positive_wave, negative_wave])))
            # calculates number of steps given step width and slab thickness
            steps = int(floor(slab_thickness / step_width))
            # adds relevant positions inside slab to the position list
            [position_list.append(starting_thickness + i * step_width) for i in range(1, steps + 1)]

            # calculates the real part of the sum of both incident and reflected wave fields
            for i in range(1, steps + 1):
                prop_mat = PropMatrix(thickness=step_width * i,
                                      wavelength=wavelength,
                                      material=material).matrix
                field = abs(sum(tuple(np.inner(prop_mat, np.array([positive_wave, negative_wave])))).real)
                field_list.append(field)

            # calculates field just before interface
            prop_mat = PropMatrix(thickness=slab_thickness,
                                  wavelength=wavelength,
                                  material=material).matrix
            field = abs(sum(tuple(np.inner(prop_mat, np.array([positive_wave, negative_wave])))).real)
            field_list.append(field)
            position_list.append(starting_thickness + slab_thickness)

            starting_thickness += slab_thickness
            positive_wave, negative_wave = tuple(np.inner(prop_mat, np.array([positive_wave, negative_wave])))

        return position_list, field_list
