import numpy as np
import cmath

from numpy.ma.core import floor
from numpy.typing import NDArray

from abstraction import Stack, Material, Slab, PropMatrix
from typing import List, Tuple

class Calc:
    def __init__(self, stack: Stack):
        self.stack = stack

    def get_r_t_from_matrix(self, wavelength):
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
        transmission = abs(t) ** 2 * n_substrate.real / n_medium.real
        return reflectance, transmission

    def get_r_t_from_series(self, wavelength):
        n_inc = self.stack.medium.get_refr_ind(wavelength)
        n_sub = self.stack.substrate.get_refr_ind(wavelength)
        n_slab = self.stack.slab_stack[0].material.get_refr_ind(wavelength)
        r_12 = (n_inc - n_slab) / (n_inc + n_slab)
        r_23 = (n_slab - n_sub) / (n_slab + n_sub)
        phi = 2 * np.pi / wavelength * n_slab * self.stack.slab_stack[0].thickness

        r = (r_12 + r_23 * np.exp(-2j * phi)) / (1 + r_12 * r_23 * np.exp(-2j * phi))

        reflection = abs(r) ** 2
        return reflection, 1 - reflection

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
            transmission = abs(t) ** 2 * n_substrate.real / n_medium.real

            transmission_values.append(transmission)
            reflectance_values.append(reflectance)

        absorption = 1 - np.array(transmission_values) - np.array(reflectance_values)

        return np.array(wavelengths), np.array(transmission_values), np.array(reflectance_values), absorption


    def calc_field_incident_medium(self, step_width: float, wavelength: float):
        position_list = np.array([])
        field_list_positive = np.array([])
        field_list_negative = np.array([])
        stack_matrix = self.stack.make_matrix(wavelength)
        incident_amplitude = 0.5
        # entries of the transfer matrix
        c = stack_matrix[1, 0]
        d = stack_matrix[1, 1]
        reflected_amplitude = -incident_amplitude * c / d

        n_inc = self.stack.medium.get_refr_ind(wavelength)
        k_inc = 2 * np.pi * n_inc / wavelength

        # recreates E-field in incident medium
        z = np.array(np.arange(0, -wavelength / 2, -step_width))
        field_positive = incident_amplitude * np.exp(-1j * k_inc * z)
        field_negative = reflected_amplitude * np.exp(1j * k_inc * z)
        position_list = np.append(position_list, z)
        field_list_positive = np.append(field_list_positive, field_positive)
        field_list_negative = np.append(field_list_negative, field_negative)

        position_list = np.flip(position_list)
        field_list_positive = np.flip(field_list_positive)
        field_list_negative = np.flip(field_list_negative)

        return position_list, field_list_positive, field_list_negative


    def make_efield_array(self, step_width: float, wavelength: float):
        position_list = np.array([])
        field_list_positive = np.array([])
        field_list_negative = np.array([])

        # recreates field inside incident medium
        incidence_data = self.calc_field_incident_medium(step_width, wavelength)

        position_list = np.append(position_list, incidence_data[0])
        field_list_positive = np.append(field_list_positive, incidence_data[1])
        field_list_negative = np.append(field_list_negative, incidence_data[2])

        # acquires a list of all interface and propagation matrices in reversed order (closest to incidence first)
        reversed_matrix_list = self.stack.make_matrix_list(wavelength)
        interface_matrices = reversed_matrix_list[::2]
        slabs: List[Slab] = [slab for slab in self.stack.slab_stack]
        slabs.reverse()

        # initialises thickness to zero
        starting_thickness = 0
        for (interface_matrix, slab) in zip(interface_matrices, slabs):
            material = slab.material
            slab_thickness = slab.thickness
            # calculates number of steps given step width and slab thickness
            steps = int(floor(slab_thickness / step_width))

            # initialises left-side field amplitudes
            positive_wave = field_list_positive[-1]
            negative_wave = field_list_negative[-1]

            # calculates amplitudes on the right side of the interface
            positive_wave, negative_wave = tuple(np.inner(interface_matrix, np.array([positive_wave, negative_wave])))

            # calculates the real part of the sum of both incident and reflected wave fields
            z = np.array(np.linspace(0, slab_thickness, steps + 1))
            n_slab = slab.material.get_refr_ind(wavelength)
            k_slab = 2 * np.pi * n_slab / wavelength
            field_positive = positive_wave * np.exp(-1j * k_slab * z)
            field_negative = negative_wave * np.exp(1j * k_slab * z)

            position_list = np.append(position_list, starting_thickness + z)
            field_list_positive = np.append(field_list_positive, field_positive)
            field_list_negative = np.append(field_list_negative, field_negative)

            starting_thickness += slab_thickness

        # calculating field inside substrate
        positive_wave = field_list_positive[-1]
        negative_wave = field_list_negative[-1]
        interface_matrix = interface_matrices[-1]
        positive_wave, negative_wave = tuple(np.inner(interface_matrix, np.array([positive_wave, negative_wave])))
        n_sub = self.stack.substrate.get_refr_ind(wavelength)
        k_sub = 2 * np.pi * n_sub / wavelength
        z = np.array(np.arange(0, wavelength / 2, step_width))
        field_positive = positive_wave * np.exp(-1j * k_sub * z)
        field_negative = negative_wave * np.exp(1j * k_sub * z)
        position_list = np.append(position_list, z + starting_thickness)
        field_list_positive = np.append(field_list_positive, field_positive)
        field_list_negative = np.append(field_list_negative, field_negative)



        return position_list, field_list_positive, field_list_negative

    def get_resulting_field(self, step_width: float, wavelength: float|None = None):
        if not wavelength:
            wavelength = self.stack.target_wavelength
        results = self.make_efield_array(step_width, wavelength)
        positive_wave = results[1]
        negative_wave = results[2]
        wavelengths = results[0]
        resulting_field = np.absolute(np.add(positive_wave, negative_wave).real)

        return wavelengths, resulting_field
