import os
import sys
from aerosandbox.geometry import Airplane, Wing, WingXSec, Airfoil
from aerosandbox.library.airfoils import naca0008
from aerosandbox.geometry.common import cosspace
from aerosandbox.aerodynamics import Casvlm1
from aerosandbox.performance import OperatingPoint
import casadi as cas
import numpy as np


def naca_4(m, p, t, n_points_per_side=100):
    # https://en.wikipedia.org/wiki/NACA_airfoil#Four-digit_series

    # Make uncambered coordinates
    x_t = cosspace(0, 1, n_points_per_side)  # Generate some cosine-spaced points
    y_t = 5 * t * (
            + 0.2969 * x_t ** 0.5
            - 0.1260 * x_t
            - 0.3516 * x_t ** 2
            + 0.2843 * x_t ** 3
            - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
    )

    if p == 0:
        p = 0.5  # prevents divide by zero errors for things like naca0012's.

    # Get camber
    y_c = cas.if_else(
        x_t <= p,
        m / p ** 2 * (2 * p * x_t - x_t ** 2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x_t - x_t ** 2)
    )

    # Get camber slope
    dycdx = cas.if_else(
        x_t <= p,
        2 * m / p ** 2 * (p - x_t),
        2 * m / (1 - p) ** 2 * (p - x_t)
    )
    theta = cas.atan(dycdx)

    # Combine everything
    x_u = x_t - y_t * cas.sin(theta)
    x_l = x_t + y_t * cas.sin(theta)
    y_u = y_c + y_t * cas.cos(theta)
    y_l = y_c - y_t * cas.cos(theta)

    # Flip upper surface so it's back to front
    x_u, y_u = x_u[::-1, :], y_u[::-1, :]

    # Trim 1 point from lower surface so there's no overlap
    x_l, y_l = x_l[1:], y_l[1:]

    x = cas.vertcat(x_u, x_l)
    y = cas.vertcat(y_u, y_l)

    coordinates = np.array(cas.horzcat(x, y))

    return coordinates


class HiddenPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self._original_stdout


class GliderSimulator:
    def __init__(self):
        self.opti = cas.Opti()  # Initialize an analysis/optimization environment
        self.opti.solver('ipopt')
        self.design_history = []
        self.output_history = []

        self.weight = 2 * 9.81  # Weight corresponding to 2 kg

        # Define the 3D geometry you want to analyze/optimize.
        # All distances are in meters and all angles are in degrees.

        self.airplane = Airplane(
            name="Glider",
            x_ref=0,  # CG location
            y_ref=0,  # CG location
            z_ref=0,  # CG location
            wings=[
                Wing(
                    name="Main Wing",
                    x_le=0,  # Coordinates of the wing's leading edge
                    y_le=0,  # Coordinates of the wing's leading edge
                    z_le=0,  # Coordinates of the wing's leading edge
                    symmetric=True,
                    xsecs=[  # The wing's cross ("X") sections
                        # Airfoils are blended between a given XSec and the next one.
                        WingXSec(  # Root
                            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                            chord=0.18,
                            twist=0,  # degrees
                            airfoil=Airfoil(coordinates=naca_4(0.04, 0.4, 0.12)),
                            control_surface_type='symmetric',
                            # Flap # Control surfaces are applied between a given XSec and the next one.
                            control_surface_deflection=0,  # degrees
                        ),
                        WingXSec(  # Mid 1
                            x_le=0.01,
                            y_le=0.33,
                            z_le=0,
                            chord=0.18,
                            twist=0.0,
                            airfoil=Airfoil(coordinates=naca_4(0.04, 0.4, 0.12)),
                            control_surface_type='asymmetric',  # Aileron
                            control_surface_deflection=0,
                        ),
                        WingXSec(  # Mid 2
                            x_le=0.02,
                            y_le=0.66,
                            z_le=0,
                            chord=0.18,
                            twist=-0.0,
                            airfoil=Airfoil(coordinates=naca_4(0.04, 0.4, 0.12)),
                            control_surface_type='asymmetric',  # Aileron
                            control_surface_deflection=0,
                        ),
                        WingXSec(  # Tip
                            x_le=0.08,
                            y_le=1,
                            z_le=0.0,
                            chord=0.18,
                            twist=0,
                            airfoil=Airfoil(coordinates=naca_4(0.04, 0.4, 0.12)),
                        ),
                    ]
                ),
            ]
        )

        self.ap = Casvlm1(  # Set up the AeroProblem
            airplane=self.airplane,
            op_point=OperatingPoint(
                density=1.225,  # kg/m^3
                viscosity=1.81e-5,  # kg/m-s
                velocity=10,  # m/s
                mach=0,  # Freestream mach number
                alpha=5,  # In degrees
                beta=0,  # In degrees
                p=0,  # About the body x-axis, in rad/sec
                q=0,  # About the body y-axis, in rad/sec
                r=0,  # About the body z-axis, in rad/sec
            ),
            opti=self.opti,  # Pass it an optimization environment to work in
            run_setup=False
        )

    def modify(self, x):
        """Modifies a list of design variables of the airplane to the elements of x."""
        tmp = self.airplane.wings[0].xsecs[1]
        # tmp.y_le = x[0]
        # self.airplane.wings[0].xsecs[1].xyz_le = cas.vertcat(tmp.x_le, x[0], tmp.z_le)
        tmp.chord = x[0]

        tmp = self.airplane.wings[0].xsecs[2]
        tmp.chord = x[1]

        tmp = self.airplane.wings[0].xsecs[3]
        tmp.chord = x[2]

        self.ap.setup(verbose=False)

    def simulate(self, x):
        for i in range(len(self.design_history)):
            if np.all(x == self.design_history[i]):
                return self.output_history[i]

        self.design_history.append(x)
        with HiddenPrints(True):
            self.modify(x)
            sol = self.opti.solve()
        cm = sol.value(self.ap.Cm)
        lift = sol.value(self.ap.lift_force)
        drag = sol.value(self.ap.drag_force_induced)
        self.output_history.append((lift, drag, cm))
        self.ap.substitute_solution(sol)

        return lift, drag, cm

    def airfoil_area(self, x):
        return Airfoil(coordinates=naca_4(x[0], x[1], x[2])).area()

    def airfoil_ixx(self, x):
        return Airfoil(coordinates=naca_4(x[0], x[1], x[2])).Ixx()

    def draw(self):
        self.ap.draw()

    def aspect_ratio(self):
        ar = self.airplane.wings[0].aspect_ratio()
        return float(ar)


if __name__ == '__main__':
    simulator = GliderSimulator()
    output = simulator.simulate((0.18, 0.18, 0.18))
    simulator.draw()
