from functools import reduce

def add(a, b):
    return a + b

def mean(iter):
    return reduce(add, iter)/len(iter)