import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution, dual_annealing
from examples.Project_2.glidersimulator2 import GliderSimulator
from examples.optimtools import Simulator
from pyDOE2 import fullfact

glider_simulator = GliderSimulator()
x0 = [0.2, 0.15, 0.18]
names = ["y_mid_1", "chord_mid_1", "chord_mid_2", "chord_mid_3"]


# Full factorial first
levels = 10

F = np.array((np.linspace(0.1, 0.60, levels),
              np.linspace(0.05, 0.20, levels),
              np.linspace(0.05, 0.20, levels),
              np.linspace(0.05, 0.20, levels)))
n = F.shape[0]
le = fullfact(n*[levels]).astype(int)
o = levels**n
FF = np.zeros((o, n+4))
for i in range(o):
    for j in range(n):
        FF[i, j] = F[j, le[i, j]]

if Path("FF.pickle").exists():
    with open("FF.pickle", "rb") as f:
        FF = pickle.load(f)
else:
    timing = [time.perf_counter()]
    for i in range(o):
        print(f"{i + 1}/{o}", end='\r')
        FF[i, -3:] = glider_simulator.simulate(FF[i, :])
        timing.append(time.perf_counter())

    with open("timings.pickle", "wb") as f:
        pickle.dump(timing, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("FF.pickle", "wb") as f:
        pickle.dump(FF, f, protocol=pickle.HIGHEST_PROTOCOL)

# Main effects
obj = FF[:, 5]
# obj = FF[:, 4]
m = np.mean(obj)
M = np.zeros((n, levels))

for j in range(n):
    for l in range(levels):
        indices = np.where(FF[:, j] == F[j, l])
        M[j, l] = np.mean(obj[indices]) - m
    plt.figure(names[j])
    plt.plot(F[j, :], M[j, :])

M_effects = np.diff(M, axis=1)
for j in range(n):
    plt.figure(f"{names[j]}_effect")
    plt.bar((F[0, 1:] + F [0, :-1]) / 2, M_effects[j, :], width=np.diff(F[0, :]) / 2)

plt.figure("Average effect over range")
plt.bar(names, np.mean(M_effects, axis=1))



def obj(x):
    output = glider_simulator.simulate(x)
    return output[1]


def con(x):
    output = glider_simulator.simulate(x)
    return output[0]

simulator = Simulator(obj)

print("Started optimization")


n_con = NonlinearConstraint(con, (15,), (np.inf,))
# bounds = [(0.1, 0.60),
#          (0.05, 0.20),
#          (0.05, 0.20),
#          (0.05, 0.20)]
bounds = [(0.05, 0.20),
         (0.05, 0.20),
         (0.05, 0.20)]


res_1 = minimize(simulator.simulate, x0, method='SLSQP', bounds=bounds,
               constraints=n_con, callback=simulator.callback)

simulator.reset()
res_2 = minimize(simulator.simulate, x0, method='trust-constr', bounds=bounds,
               constraints=n_con, callback=simulator.callback)


def con(x):
    output = glider_simulator.simulate(x)
    return (*x, output[0])



n_con = NonlinearConstraint(con, (0.05, 0.05, 0.05, 15), (0.20, 0.20, 0.20, np.inf))
simulator.reset()

res_3 = minimize(simulator.simulate, x0, args={'verbose': True}, method='COBYLA',
               constraints=n_con, tol=1e-6)


def obj(x):
    output = glider_simulator.simulate(x)
    if output[0] < 15:
        return np.inf
    else:
        return output[1]


simulator = Simulator(obj)

res_4 = differential_evolution(simulator.simulate, bounds, args={'verbose': True}, workers=-1)

simulator.reset()
res_5 = dual_annealing(simulator.simulate, bounds, callback=simulator.callback)
