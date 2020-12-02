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
x0 = np.array((0.19, 0.18, 0.17))
names = ["chord_mid_1", "chord_mid_2", "chord_mid_3"]

# Design of experiments full factorial
print("=== \033[1mDesign of experiments\033[0m ===")

levels = 30

F = np.array((np.linspace(0.05, 0.20, levels),
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
    plt.bar((F[0, 1:] + F[0, :-1]) / 2, M_effects[j, :], width=np.diff(F[0, :]) / 2)

plt.figure("Average effect over range")
plt.bar(names, np.mean(M_effects, axis=1))

print("=== \033[1mStarted optimization\033[0m ===")
print("\nAttempt with SLSQP method")


def obj(x):
    output = glider_simulator.simulate(x)
    return output[1]


simulator = Simulator(obj)

bounds = [(0.05, 0.20),
          (0.05, 0.20),
          (0.05, 0.20)]


def con(x):
    output = glider_simulator.simulate(x)
    return output[0]


n_con = {'type': 'ineq', 'fun': lambda x: con(x) - 15}

res_1 = minimize(simulator.simulate, x0, method='SLSQP', bounds=bounds, constraints=n_con, callback=simulator.callback,
                 tol=1e-8, options={'ftol': 1e-8})

print("\nAttempt with COBYLA method")

n_con = [{'type': 'ineq', 'fun': lambda x: x[0] - 0.05},
         {'type': 'ineq', 'fun': lambda x: x[1] - 0.05},
         {'type': 'ineq', 'fun': lambda x: x[2] - 0.05},
         {'type': 'ineq', 'fun': lambda x: 0.2 - x[0]},
         {'type': 'ineq', 'fun': lambda x: 0.2 - x[1]},
         {'type': 'ineq', 'fun': lambda x: 0.2 - x[2]},
         {'type': 'ineq', 'fun': lambda x: con(x) - 15}]

simulator.reset()

res_2 = minimize(simulator.simulate, x0, args=((), {'verbose': True},), method='COBYLA', constraints=n_con, tol=1e-8,
                 options={'rhobeg': 1e-4})

print("\nAttempt with trust-constr method")

n_con = NonlinearConstraint(con, (15,), (np.inf,))
simulator.reset()
res_3 = minimize(simulator.simulate, x0, method='trust-constr', bounds=bounds, constraints=n_con,
                 callback=simulator.callback)


print("\nAttempt with differential evolution method")


def obj_con(x):
    output = glider_simulator.simulate(x)
    if output[0] < 15:
        return np.inf
    else:
        return output[1]


simulator = Simulator(obj_con)
res_4 = differential_evolution(simulator.simulate, bounds, args=((), {'verbose': True}), workers=-1)

# print("\nAttempt with dual annealing method")  # Very slow
# simulator.reset()
# res_5 = dual_annealing(simulator.simulate, bounds, callback=simulator.callback)

print("\nAttempt with Nelder-Mead simplex method")


def obj_con_bound(x):
    if np.any(x < np.array(bounds)[:, 0]) or np.any(x > np.array(bounds)[:, 1]):
        return np.inf
    else:
        return obj(x)


simulator = Simulator(obj_con_bound)
res_6 = minimize(simulator.simulate, x0, method="Nelder-mead", callback=simulator.callback)
