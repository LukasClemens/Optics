import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation

from abstraction import material_dict, Material, Slab, PropMatrix, InterMatrix, Stack
from calc import Calc

class Visualiser:
    def __init__(self, calc: Calc):
        self.calc = calc

    def show_spectrum(self, start_wl, stop_wl, step_wl=1, show_max_r=False):
        wavelengths, t_spec, r_spec, absorption = self.calc.make_spectrum_array(start_wl, stop_wl, step_wl)

        fig, ax = plt.subplots(figsize=(12.5, 5))
        ax.set_xlabel('$wavelength$ $[nm]$')
        ax.set_title('Transmission and Reflection spectra for the single slab AR coating')
        if self.calc.stack.dispersive:
            ax.set_ylabel('$T/R/A$')
        else:
            ax.set_ylabel('T/R')

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(visible=True, which='both', axis='both')

        t_line = ax.plot(wavelengths, t_spec, lw=1, label='$Transmission$', color='k')
        r_line = ax.plot(wavelengths, r_spec, lw=1, label='$Reflection$', color='b')
        if self.calc.stack.dispersive:
            a_line = ax.plot(wavelengths, absorption, lw=1, label='$Absorption$', color='r')


        if show_max_r:
            index_design_wl = np.where(wavelengths == self.calc.stack.target_wavelength)[0][0]
            r_design_wl = round(r_spec[index_design_wl], 5)
            r_percent = r_design_wl * 100

            bbox_style = dict(boxstyle='round', fc='0.8')
            arrowprops = dict(arrowstyle='->')

            ax.annotate(text='$R_{700nm}=$ %1.3f'%r_percent + '%',
                        xy=(self.calc.stack.target_wavelength, r_design_wl),
                        xytext=(self.calc.stack.target_wavelength, r_design_wl - 0.15),
                        arrowprops=arrowprops,
                        bbox=bbox_style)

        ax.legend()
        plt.show()

        return

    def show_efield(self, step_width: float, wavelength: float|None=None):
        z, field = self.calc.get_resulting_field(step_width, wavelength)

        # makes a list of all interface positions
        initial_postion = 0
        interface_positions = []
        for slab in self.calc.stack.slab_stack:
            interface_positions.append(initial_postion)
            initial_postion += slab.thickness
        interface_positions.append(initial_postion)

        fig, ax = plt.subplots(figsize=(12.5, 5))
        ax.set_title('E-field distribution inside the symmetric design $(HL)^{15}H$')
        ax.set_xlabel('$z$ $[nm]$')
        ax.set_ylabel('Relative electric field strength')
        ax.grid(visible=True, which='major', axis='y')

        ax.plot(z, field, lw=1, color='r')
        ax.vlines(interface_positions, ymin=0, ymax=0.5, colors='k', linestyles='dotted', zorder=0)

        plt.show()

        return

    def animate_efield(self, step_width: float, wavelength: float|None=None):
        if not wavelength:
            wavelength = self.calc.stack.target_wavelength

        omega = 2 * np.pi * 299792458 / (wavelength * 10 ** -9)
        t_period = 2 * np.pi / omega
        frames = 120
        duration = 2
        t_step = t_period / frames
        frame_duration = duration / frames * 1000
        z, e_pos, e_neg = self.calc.make_efield_array(step_width, wavelength)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title('E-field distribution inside and outside the slab')
        ax.set_xlabel('$z$ $[nm]$')
        ax.set_ylabel('Relative electric field strength')
        ax.grid(visible=True, which='major', axis='y')
        ax.set_xlim(-500, 2500)
        ax.set_ylim(-3, 3)

        line_res = ax.plot([], [], color='k', alpha=0.5, label='Resulting field')[0]
        line_pos = ax.plot([], [], color='r', label='Wave in positive direction')[0]
        line_neg = ax.plot([], [], color='b', label='Wave in negative direction')[0]

        initial_position = 0
        interface_positions = []
        for slab in self.calc.stack.slab_stack:
            interface_positions.append(initial_position)
            initial_position += slab.thickness
        interface_positions.append(initial_position)
        ax.vlines(interface_positions, ymin=-3, ymax=3, colors='k', linestyles='dotted', zorder=0)

        ax.legend(loc='lower center')

        def update(i):
            e_pos_new = e_pos * np.exp(1j * omega * i * t_step)
            e_neg_new = e_neg * np.exp(1j * omega * i * t_step)
            e_res = e_pos_new + e_neg_new

            line_pos.set_data(z, e_pos_new.real)
            line_neg.set_data(z, e_neg_new.real)
            line_res.set_data(z, e_res.real)
            return line_pos, line_neg, line_res

        ani = FuncAnimation(fig=fig, func=update, frames=frames, interval=frame_duration)
        # ani.save(filename='results/animation_cavity.gif', writer='pillow')
        plt.show()


