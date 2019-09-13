import numpy as np
from scipy import stats 
import mixfit

def test(params, n):
    tau, mu1, sigma1, mu2, sigma2 = params
    x_n1 = stats.norm.rvs(loc=mu1, scale=sigma1, size=int(tau*n))
    x_n2 = stats.norm.rvs(loc=mu2, scale=sigma2, size=int((1-tau)*n))
    x = np.concatenate((x_n1, x_n2))
    tau, mu1, sigma1, mu2, sigma2 = params
    result1 = mixfit.max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-2)
    params1 = np.around(result1, 2)
    tau, mu1, sigma1, mu2, sigma2 = params
    result2 = mixfit.em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-2)
    params2 = np.around(result2, 2)
    return params1, params2

def main():
    params = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    n = 10000
    params1, params2 = test(params, n)
    print ("real", params)
    print("likelihood", params1)
    print("em", params2)
    np.testing.assert_allclose(params, params1, rtol=0.1, atol=0)
    np.testing.assert_allclose(params, params2, rtol=0.1, atol=0)


if __name__ == '__main__':
    main()