import numpy as np

def func(x):
    return 5*np.exp(-x)+x-5


def binary_search(f, x1, x2):
    N = 0
    delta = x2-x1
    while delta/(2**N) > 10**-6:
        mid_x = (x2+x1)/2
        if f(mid_x)*f(x1)<0:
            x2 = mid_x
        else:
            x1 = mid_x
        N+=1
        print(mid_x)
    return mid_x

print(binary_search(func, 2, 10))