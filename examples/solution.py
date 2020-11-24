import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from examples.glidersimulator import GliderSimulator
from examples.optimtools import Simulator

from pyDOE2 import fullfact

glider_simulator = GliderSimulator()
output = glider_simulator.simulate((0.04, 0.4, 0.12, 5))
names = ["NACA_m", "NACA_p", "NACA_t", "alpha"]

# Full factorial first
levels = 10

F = np.array((np.linspace(0, 0.09, levels),
              np.linspace(0, 0.90, levels),
              np.linspace(0, 0.99, levels),
              np.linspace(-3, 15, levels)))
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
        FF[i, -4:] = glider_simulator.simulate(FF[i, :])
        timing.append(time.perf_counter())

    with open("timings.pickle", "wb") as f:
        pickle.dump(timing, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("FF.pickle", "wb") as f:
        pickle.dump(FF, f, protocol=pickle.HIGHEST_PROTOCOL)

# Main effects
obj = FF[:, 4] / FF[:, 5]
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

# Constraint area > 0
# areas = np.zeros(o)
# Ixx = np.zeros(o)
# for i in range(len(areas)):
#     areas[i] = simulator.area(FF[i, :3])
#     Ixx[i] = simulator.Ixx(FF[i, :3])
#
# mask = areas > 0.05
# mask = np.greater(Ixx, 0.01, where=np.nan)
# mask = FF[ :, 2] > 0
# x_start = FF[mask][np.argmax(obj[mask]), :4]


# Optimize for Cl over Cd
# Gradient

def obj(x):
    output = glider_simulator.simulate(x)
    return -output[0]/output[1] + 10000000*output[2]**2


simulator = Simulator(obj)


print("Started optimization")
bounds = [(0, 0.09),
         (0, 0.90),
         (0, 0.99),
         (-3, 15)]
res = minimize(simulator.simulate, np.array([0.04, 0.4, 0.12, 5]), method='L-BFGS-B', bounds=bounds,
               callback=simulator.callback)

# Non-gradient



# Optimize for v
# Optimize for Cd