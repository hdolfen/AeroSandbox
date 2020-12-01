import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, root_scalar
from examples.Project_1.glidersimulator import GliderSimulator
from examples.optimtools import Simulator

glider_simulator = GliderSimulator()
x0 = [0.04, 0.4, 0.5, 8, -10]


def obj(x):
    output = glider_simulator.simulate(x)
    return output[1]


def con(x):
    output = glider_simulator.simulate(x)
    return output[0], output[2]


res = root_scalar(lambda x: con([*x0[:4], x])[1], method='toms748', bracket=(-20, 20))

x0[4] = res.root

simulator = Simulator(obj)

print("Started optimization")
n_con = NonlinearConstraint(con, (17, -1e-5), (np.inf, 1e-5))
bounds = [(0, 0.09),
         (0, 0.90),
         (0.01, 0.99),
         (-3, 15),
          (-20, 20)]

res = minimize(simulator.simulate, x0, method='trust-constr', bounds=bounds,
               constraints=n_con, callback=simulator.callback)
