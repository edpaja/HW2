import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def func(x):
    return (x[0]-2)**2+(x[1]-2)**2

#central difference method in N dimensions
def N_gradient(f, x, h=1**-6):
    x = np.asarray(x, float) #x vector
    g = np.zeros_like(x) #empty vector of same shape as x for gradient vector
    for k in range(x.size): #iterate components of x
        xf, xb = x.copy(), x.copy()
        xf[k] += h; xb[k] -= h
        g[k] = (f(xf) - f(xb))/(2*h)
    return g #returns approx gradient of f at x

def gradient_descent(f, x_init, max_iter = 100000,tol=1**-5, h = 1e-3):#track gradient descent for a function f starting at x_init
    x = np.asarray(x_init, float) #x vector
    path = [x]
    for i in range(max_iter):#EXPLAIN MAX ITER
        grad = N_gradient(f, x)
        step = np.sqrt(grad@grad)
        x = x - grad*(h/step)
        path.append(x)
    return x, np.array(path)

minimum = [2, 2]
print(N_gradient(func, (0,0)))
print(N_gradient(func, (2,2)))
print(gradient_descent(func, (0,0)))

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
X, Y = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-4, 4, 400))
ax.contour(X, Y, func([X, Y]))
for s in [(0, 0), (-1, 1),(4, -3)]:
    xb, path = gradient_descent(func, s)
    ax.plot(path[:, 0], path[:, 1], ".-", ms=3, lw=1, label=f"start {s}")
ax.set(xlabel="x", ylabel="y", title="Paths to function minimum using different initial x")
ax.legend(fontsize=7)
ax.legend(); plt.tight_layout(); plt.savefig("test_function.png", dpi=130); plt.close()