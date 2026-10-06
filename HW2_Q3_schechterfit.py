import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#central difference method in N dimensions
def N_gradient(f, x, h=1e-6):
    x = np.asarray(x, float) #x vector
    g = np.zeros_like(x) #empty vector of same shape as x for gradient vector
    for k in range(x.size): #iterate components of x
        xf, xb = x.copy(), x.copy()
        xf[k] += h; xb[k] -= h
        g[k] = (f(xf) - f(xb))/(2*h)
    return g #returns approx gradient of f at x


def gradient_descent(f, x_init, max_iter = 10000,tol=1**-5):#track gradient descent for a function f starting at x_init
    x = np.asarray(x_init, float) #x vector
    path = [x]
    fg = f(x)
    ftrail = [fg]
    step = 1.0
    for i in range(max_iter):#EXPLAIN MAX ITER
        grad = N_gradient(f, x)
        g2 = grad@grad
        if np.sqrt(g2) < 1e-10: #stop as we have reached the minimum to machine precision
            break
        step = min(step * 1.5, 1e6) #double 
        step = min(step, 0.01/np.sqrt(g2)) #cap at the 
        while True:                        
            xn = x-step*grad
            fn = f(xn)
            if 1e-4*g2*step <= fg - fn: #if the gradient increases our f(x) value(increases instead of descending)
                break
            step = step*0.5
            if step < 1e-300:
                return x, np.array(path), np.array(ftrail)
        x, fx = xn, fn
        path.append(x.copy()); ftrail.append(fx)
    print(x)
    return x, np.array(path), np.array(ftrail)

logM, n_gal, err = np.loadtxt("/Users/edouardsavalle/Desktop/University/grad/Computational_Physics/HW2/smf_cosmos.dat", unpack=True)
M = 10**logM
 
def n(logM, phi, Ms, alpha):
    x = 10**logM / Ms
    return phi * x**(alpha + 1) * np.exp(-x) * np.log(10)
 
# Parameters searched: theta = (phi*/1e-3, log10 M*, alpha)  (rescaled so the
# parameters are all O(1); the model is still phi*, M*, alpha).
def scale(t):
    return 1e-3*t[0], 10**t[1], t[2]
def chi2(t):
    phi, Ms, a = scale(t)
    return np.sum(((n_gal - n(logM, phi, Ms, a)) / err)**2)


#rescaling and setting phi, logM*, and alpha to our result
phi, Ms, a = scale([2.70, 10.98, -1.01])

mm = np.linspace(9.0, 12.0, 300)
plt.figure(figsize=(6.5, 5))
plt.errorbar(10**logM, n_gal, err, fmt="o", capsize=3, label="data")
plt.loglog(10**mm, n(mm, phi, Ms, a), "-",
           label=rf"Schechter model optimized parameters: $\phi^*$={phi:.2e}, $\log M^*$={np.log10(Ms):.2f}, $\alpha$={a:.2f}")
plt.xlabel(r"$M_{gal}\ [M_\odot]$"); plt.ylabel(r"$n(M)$ [1/dex/(Mpc/h)$^3$]")
plt.legend(fontsize=8); plt.title(rf"Schechter best fit")
plt.tight_layout(); plt.savefig("schechter_fit.png", dpi=130)

    
