import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution, dual_annealing
from examples.Project_2.glidersimulator import GliderSimulator
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
n_con = NonlinearConstraint(con, (15,), (np.inf,))

res_1 = minimize(simulator.simulate, [0.175, 0.15, 0.09], method='SLSQP', bounds=bounds, constraints=n_con, callback=simulator.callback,
                 tol=1e-6, options={'ftol': 1e-6, 'eps': 1e-6})

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
        return obj_con(x)


simulator = Simulator(obj_con_bound)
res_6 = minimize(simulator.simulate, x0, method="Nelder-mead", callback=simulator.callback)
# Restarting it at with an initial guess from previous optimizations can help to decrease it further
res_6 = minimize(simulator.simulate, res_6.x, method="Nelder-mead", callback=simulator.callback)
res_6 = minimize(simulator.simulate, res_6.x, method="Nelder-mead", callback=simulator.callback)

print("=== \033[1mAttempt with L-BFGS-B method\033[0m ===")


def obj_pen(x, mu):
    output = glider_simulator.simulate(x)
    return output[1] + mu*min(0, output[0] - 15)**2


simulator = Simulator(lambda x: obj_pen(x, 10))
res_7 = minimize(simulator.simulate, x0, method='L-BFGS-B', bounds=bounds, callback=simulator.callback,
                 options={'eps': 1e-3, 'ftol': 1e-20, 'gtol': 1e-20})


print("=== \033[1mAttempt with TNC method\033[0m ===")
simulator.reset()
res_8 = minimize(simulator.simulate, x0, method='TNC', bounds=bounds, callback=simulator.callback)
simulator = Simulator(lambda x: obj_pen(x, 1000))
res_8 = minimize(simulator.simulate, res_8.x, method='TNC', bounds=bounds, callback=simulator.callback,
                 options={'maxiter': 300})


# Multi-objective optimization
print("=== \033[1mStarted multi-objective optimization\033[0m ===")


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


ar_list = []
history = np.vstack((glider_simulator.design_history, FF[:, :3]))
for design in history:
    ar_list.append(glider_simulator.aspect_ratio(design))

drag_list = np.concatenate((np.array(glider_simulator.output_history)[:, 1], FF[:, 5]))
lift_list = np.concatenate((np.array(glider_simulator.output_history)[:, 0], FF[:, 4]))
costs = np.array((drag_list, ar_list)).T

mask = lift_list > 15
costs = costs[mask]

plt.figure("Pareto")
plt.plot(costs[:, 0], costs[:, 1], 'bo')

pareto_front = is_pareto_efficient_simple(costs)
pareto_costs = costs[pareto_front]
plt.plot(pareto_costs[:, 0], pareto_costs[:, 1], 'ro')

weights = pareto_costs[:, 0] / pareto_costs[:, 1]


def obj(x, weights=(1, 1)):
    obj_1 = glider_simulator.simulate(x)
    obj_2 = glider_simulator.aspect_ratio(x)
    return weights[0]*obj_1[1] + weights[1]*obj_2


weight_space = np.linspace(0, 1, 8)
zipped_weights = np.array((np.flip(weight_space), weight_space)).T  # First row is most weight to drag
zipped_weights[:, 0] *= 25  # Magic number, base on DoE of both objectives

# weight_space = np.logspace(np.min(weights), np.max(weights), 8)
# zipped_weights = np.ones((len(weight_space), 2))
# zipped_weights[:, 1] = weight_space
# zipped_weights = np.vstack(((1, 0), zipped_weights, (0, 1)))

x0 = res_3.x
results = []
for i, row in enumerate(zipped_weights):
    print(f"\nSimulation {i + 1} with weights {row}")
    simulator = Simulator(lambda x: obj(x, weights=row))
    res = minimize(simulator.simulate, x0, method='trust-constr', bounds=bounds, constraints=n_con,
                   callback=simulator.callback, options={'xtol': 1e-3, 'gtol': 1e-08})
    results.append(res.x)
    x0 = res.x


ar_list_pareto = []
drag_list_pareto = []
for design in results:
    ar_list_pareto.append(glider_simulator.aspect_ratio(design))
    drag_list_pareto.append(glider_simulator.simulate(design)[1])

plt.figure("Pareto")
plt.plot(drag_list_pareto, ar_list_pareto, 'go')
