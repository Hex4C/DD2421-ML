import numpy , random , math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def objective():
    pass

def create_data():
    classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    
    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
         -numpy.ones(classB.shape[0])))
    
    N = inputs.shape[0] # Number of rows (samples)
    
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    

# HINT: Use list comprehension
# a = [ expr for x in seq ]

start = None
B = None
XC = None

ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']