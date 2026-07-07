import numpy as np
import cmath
from numpy.typing import NDArray

material_dict = {
    0: (False, 1, ''),
    1: (False, 1.6, ''),
    2: (False, 2.2, ''),
    3: (True, 0, 'disp_data/Ag.txt'),
    4: (False, 1.45, ''),
    5: (True, 0, 'disp_data/test.txt')
}

class Material:
    """
    Abstraction class to make handling material properties more intuitive.
    """
    def __init__(self, dispersive: bool, refr_ind: float|complex|None = None, disp_path: str = None):
        self.dispersive = dispersive
        if self.dispersive:
            self.disp_path = disp_path
            self.dispersion_ini()
        else:
            self.refr_ind: float|complex|NDArray = refr_ind

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
            k1 = self.refr_ind[2][index1]
            k2 = self.refr_ind[2][index2]

            n = (refr_ind2 - refr_ind1) / (wavelength2 - wavelength1) * (wavelength - wavelength1) + refr_ind1
            k = (k2 - k1) / (wavelength2 - wavelength1) * (wavelength - wavelength1) + k1

            return complex(n, k)

    def dispersion_ini(self):
        self.refr_ind: float|complex|NDArray = np.loadtxt(self.disp_path, skiprows=1, delimiter='\t').T

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
    Abstraction class to make handling layer stack properties more intuitive. The first entry in the slab stack is the
    slab closest to the substrate.
    """
    def __init__(self, medium: Material, substrate: Material, stack_file: str, target_wavelength: float = 700):
        self.medium = medium
        self.substrate = substrate
        self.stack_file = stack_file
        self.slab_stack = []
        self.target_wavelength = target_wavelength
        self.make_stack()

    def make_stack(self):
        material_array = np.loadtxt(self.stack_file, skiprows=1, delimiter='\t')
        for entry in material_array:
            dispersive = material_dict[int(entry[0])][0]
            if not dispersive:
                material = Material(dispersive=dispersive, refr_ind=material_dict[int(entry[0])][1])
            else:
                material = Material(dispersive=dispersive, disp_path=material_dict[int(entry[0])][2])
            abs_thickness = entry[1] * self.target_wavelength / (4 * material.get_refr_ind(self.target_wavelength).real)
            self.slab_stack.append(Slab(material, abs_thickness))
        return
