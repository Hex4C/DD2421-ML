import numpy , random , math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def objective():
    pass


start = None
B = None
XC = None

ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']