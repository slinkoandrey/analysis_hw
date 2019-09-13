import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from collections import namedtuple
import json
import opt 

Result = namedtuple(('Result'), ('nfev', 'cost', 'gradnorm', 'x')) 

def d(z, H, omg):
    before = lambda _z: ((1-omg)*((1+_z)**3) + omg)**(-0.5)
    after = np.fromiter((integrate.quad(before, 0, t)[0] for t in z), float)
    return 3e11 * (1 + z) * after / H

def mu(z, H, omg):
    return 5*np.log10(d(z, H, omg)) - 5

def j(z, H, omg): 
    j = np.empty((z.size, 2), float)
    before = lambda _z: (1-(1+_z)**3) / (((1-omg) * ((1+_z)**3)+omg)**(3/2))
    after = np.fromiter((integrate.quad(before, 0, t)[0] for t in z), float)
    j[:, 0] = -5/(np.log(10)*H) 
    j[:, 1] = (-2.5) * (1/np.log(10)) * (3e11*(1+z) / (d(z,H,omg)*H)) * after 
    return j

def main():
    data = np.genfromtxt('jla_mub.txt', names = ('z', 'mu'))
    H_0 = 50.0
    omg_0 = 0.5
    gauss = opt.gauss_newton(
            data['mu'],
            lambda *x: mu(data['z'], *x),  
            lambda *x: j(data['z'], *x),
            (H_0, omg_0),
            k = 1, 
            tol = 1e-4)
    lm = opt.lm(
            data['mu'],
            lambda *x: mu(data['z'], *x),  
            lambda *x: j(data['z'], *x),
            (H_0, omg_0),
            lmbd0 = 1e-2,
            nu = 2,
            tol = 1e-4)
    plt.xlabel('z')
    plt.ylabel('mu')
    plt.plot(data['z'], data['mu'], 'x', label = 'data')
    plt.plot(data['z'], mu(data['z'], *gauss.x), label = 'gauss', lw = 4)
    plt.plot(data['z'], mu(data['z'], *lm.x), label = 'lm')
    plt.grid()
    plt.legend (loc ='best')
    plt.savefig('mu-z.png')
    plt.figure(2)
    plt.xlabel('cost')
    plt.ylabel('i')
    plt.plot(np.arange(len(gauss.cost)), gauss.cost, label = 'gauss')
    plt.plot(np.arange(len(lm.cost)), lm.cost, label = 'lm')
    plt.grid()
    plt.legend (loc ='best')
    plt.savefig('cost.png')
    js = {"Gauss-Newton": {"H0": gauss.x[0], "Omega": gauss.x[1], "nfev": gauss.nfev},
            "Levenberg-Marquardt": {"H0": lm.x[0], "Omega": lm.x[1], "nfev": lm.nfev}}
    with open('parameters.json', 'w') as f:
        json.dump(js, f)
    with open('parameters.json') as f:
        print(f.read())
    
if __name__ == '__main__':
    main()