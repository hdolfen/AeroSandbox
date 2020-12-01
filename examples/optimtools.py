# -*- coding: utf-8 -*-
"""
Tools to facilitate optimizing.
Contains:
    Simulator: Wraps around callable that you pass as argument when initiating
    and instance of this class. Keeps track of each function call and stores
    them in different lists.
"""

import numpy as np

class Simulator:
    """Unfortunately, the options for verbose output and retaining intermediate
    results are often quite limited in scipy. This we solve by writing a
    wrapper for the objective function and a callback function."""

    def __init__(self, function):
        self.f = function
        self.num_calls = 0
        self.callback_count = 0 # counts the number of times callback has been called, this also measures the iteration count
        self.list_calls_inp = []
        self.list_calls_res = []
        self.decreasing_list_calls_inp = []
        self.decreasing_list_calls_res = []
        self.list_callback_inp = []
        self.list_callback_res = []

    def simulate(self, x, args=(), verbose=False):
        """Executes the actual simulation and returns the result, while
        updating the lists too. Pass to optimizer without arguments or
        parentheses."""
        result = self.f(x, *args) # the actual evaluation of the function
        if not self.num_calls:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
            self.list_callback_inp.append(x)
            self.list_callback_res.append(result)
        elif result < self.decreasing_list_calls_res[-1]:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
        self.list_calls_inp.append(x)
        self.list_calls_res.append(result)
        self.num_calls += 1
        if verbose:
            s1 = ""
            for comp in x:
                s1 += f"{comp:10.5e}\t"
            s1 += f"{result:10.5e}"
            if not self.num_calls:
                s0 = ""
                for j, _ in enumerate(x):
                    tmp = f"Comp-{j + 1}"
                    s0 += f"{tmp:10s}\t"
                s0 += "Objective"
                print(s0)
            print(s1)

        return result

    def __str__(self):
        if self.num_calls:
            s = f"This simulator has been called {self.num_calls} times.\n"
            s += "The current best design is: " + str(self.decreasing_list_calls_inp[-1]) + "\n"
            s += "The current minimal value is: " + str(self.decreasing_list_calls_res[-1]) + "\n"
        else:
            s = "This simulator has been called 0 times.\n"
        return s

    def reset(self):
        """Resets all attributes of a Simulator object to their initial values.
        """
        self.num_calls = 0
        self.callback_count = 0
        self.list_calls_inp = []
        self.list_calls_res = []
        self.decreasing_list_calls_inp = []
        self.decreasing_list_calls_res = []
        self.list_callback_inp = []
        self.list_callback_res = []

    def callback(self, xk, *args, **kwargs):
        """Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses."""
        s1 = ""
        xk = np.atleast_1d(xk)
        for _i, x in reversed(list(enumerate(self.list_calls_inp))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                index = _i
                break

        for comp in xk:
            s1 += f"{comp:10.5e}\t"
        s1 += f"{self.list_calls_res[index]:10.5e}"

        self.list_callback_inp.append(xk)
        self.list_callback_res.append(self.list_calls_res[index])

        if not self.callback_count:
            s0 = ""
            for j, _ in enumerate(xk):
                tmp = f"Comp-{j+1}"
                s0 += f"{tmp:10s}\t"
            s0 += "Objective"
            print(s0)
        print(s1)
        self.callback_count += 1
