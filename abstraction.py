import numpy as np
import cmath
from numpy.typing import NDArray

class Material:
    """
    Abstraction class to make handling material properties more intuitive.
    """
    def __init__(self, dispersive: bool, refr_ind: float|complex|NDArray):
        self.dispersive = dispersive
        self.refr_ind = refr_ind

    def get_refr_ind(self, wavelength: float):
        """
        If the material is dispersive this function returns a refractive index that is interpolated from existing
        dispersion data and corresponds to the provided wavelength.
        :param wavelength:
        :return: refractive index at given wavelength
        """
        if not self.dispersive:
            # in case when material is not dispersive
            return self.refr_ind
        elif wavelength in self.refr_ind[0]:
            # in case the exact wavelength is in the existing dispersion data
            index = np.where(self.refr_ind[0]==wavelength)[0][0]
            return self.refr_ind[1][index]
        else:
            # linear interpolation
            index1 = max(np.where(self.refr_ind[0]<wavelength)[0])
            index2 = min(np.where(self.refr_ind[0]>wavelength)[0])
            wavelength1 = self.refr_ind[0][index1]
            wavelength2 = self.refr_ind[0][index2]
            refr_ind1 = self.refr_ind[1][index1]
            refr_ind2 = self.refr_ind[1][index2]

            return (refr_ind2 - refr_ind1) / (wavelength2 - wavelength1) * (wavelength - wavelength1) + refr_ind1

class Slab:
    """
    Abstraction class to make handling slab properties more intuitive.
    """
    def __init__(self, material: Material, abs_thickness: float):
        self.material = material
        self.thickness = abs_thickness

class PropMatrix:
    """
    Abstraction class to make handling propagation matrix properties more intuitive.
    """
    def __init__(self, thickness: float, wavelength: float, material: Material):
        # absolute thickness in nm
        self.thickness = thickness
        # vacuum wavelength
        self.wavelength = wavelength
        # propagation medium
        self.material = material
        # refractive index at given wavelength
        self.refr_ind = material.get_refr_ind(self.wavelength)
        # k_number inside material
        self.k_number = self.refr_ind * 2 * cmath.pi / self.wavelength
        # propagation matrix
        self.matrix = np.array([[cmath.exp(-1j * self.k_number * self.thickness), 0],
                               [0, cmath.exp(1j * self.k_number * self.thickness)]])

class InterMatrix:
    """
    Abstraction class to make handling interface matrix properties more intuitive.
    """
    def __init__(self, material1: Material, material2: Material, wavelength: float):
        self.material1 = material1
        self.material2 = material2
        self.wavelength = wavelength
        n1 = material1.get_refr_ind(self.wavelength)
        n2 = material2.get_refr_ind(self.wavelength)
        self.matrix = 1 / (2 * n2) * np.array([[n2 + n1, n2 - n1], [n2 - n1, n2 + n1]])

class Stack:
    """
    Abstraction class to make handling layer stack properties more intuitive.
    """
    def __init__(self, medium: Material, substrate: Material, stack_file: str):
        self.medium = medium
        self.substrate = substrate
        self.stack_file = stack_file
        self.make_stack()

    def make_stack(self):
        return
