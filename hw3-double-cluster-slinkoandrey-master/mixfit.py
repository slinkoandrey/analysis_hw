import numpy as np
from scipy import optimize, stats
        
def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol = 1e-3): 
    params = [tau, mu1, sigma1, mu2, sigma2]
    def likelyhood(params1):
        tau, mu1, sigma1, mu2, sigma2 = params1
        p1 = stats.norm.pdf(x, mu1, sigma1) 
        p2 = stats.norm.pdf(x, mu2, sigma2) 
        return -np.sum(np.log(tau * p1 + (1 - tau) * p2))
    result = optimize.minimize(likelyhood, x0 = params, tol = 1e-3)
    return result.x

def t_ij(x, tau, mu1, sigma1, mu2, sigma2):
    p1 = stats.norm.pdf(x, loc = mu1, scale = sigma1)
    p2 = stats.norm.pdf(x, loc = mu2, scale = sigma2)
    p = tau * p1 + (1-tau) * p2
    t1 = tau * p1 / p
    t2 = (1-tau) * p2 / p
    return t1, t2

def update_theta(x, tau, mu1, sigma1, mu2, sigma2):
    t1, t2 = t_ij(x, tau, mu1, sigma1, mu2, sigma2)
    tau = np.sum(t1) / x.size
    mu1 = np.sum(t1 * x) /  np.sum(t1)
    mu2 = np.sum(t2 * x) /  np.sum(t2)
    sigma1 = np.sqrt(np.sum(t1 * (x - mu1)**2) /  np.sum(t1))
    sigma2 = np.sqrt(np.sum(t2 * (x - mu2)**2) /  np.sum(t2))
    return tau, mu1, sigma1, mu2, sigma2

def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    new = (tau, mu1, sigma1, mu2, sigma2)
    while True:
        old = new
        new = update_theta(x, *old)
        if np.allclose(new, old, rtol = rtol, atol = 0):
            break
    return new

def t_ij_cluster(x, uniform_dens, tau1, tau2, mu1, mu2, sigma1, sigma2):
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    sigma_1 = np.diag(np.array([sigma1, sigma1]))
    sigma_2 = np.diag(np.array([sigma2, sigma2]))
    p_1 = stats.multivariate_normal.pdf(x, mu1, sigma_1)
    p_2 = stats.multivariate_normal.pdf(x, mu2, sigma_2)
    p_un = np.zeros_like(p_1) + uniform_dens
    p = tau1 * p_1 + tau2 * p_2 + (1-tau1-tau2) * p_un
    t_1 = tau1 * p_1 / p
    t_2 = tau2 * p_2 / p
    t_un = (1 - tau1 - tau2) * p_un / p 
    return t_1, t_2, t_un

def update_theta_cluster(x, tau1, tau2, mu1, mu2, sigma1, sigma2):
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    t1, t2, t_u = t_ij_cluster(x, tau1, tau2, mu1, mu2, sigma1, sigma2)
    tau1 = np.sum(t1) / (np.sum(t1) + np.sum(t_u) + np.sum(t2))
    tau2 = np.sum(t2) / (np.sum(t1) + np.sum(t_u) + np.sum(t2))
    mu1[0] = np.sum(t1 * x[:,0]) /  np.sum(t1)
    mu1[1] = np.sum(t1 * x[:,1]) /  np.sum(t1)
    mu2[0] = np.sum(t2 * x[:,0]) /  np.sum(t2)
    mu2[1] = np.sum(t2 * x[:,1]) /  np.sum(t2)
    sigma1 = np.sqrt(np.sum(t1 * (x[:,0] - mu1) ** 2) /  np.sum(t1))
    sigma2 = np.sqrt(np.sum(t1 * (x[:,1] - mu1) ** 2) /  np.sum(t1))
    return tau1, tau2, mu1, sigma1, mu2, sigma2

def em_double_cluster(x, uniform_dens, tau1, tau2, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    new = (tau1, tau2, mu1, sigma1, mu2, sigma2)
    while True:
        old = new
        new = update_theta(x, uniform_dens, *old)
        if ((np.isclose(new[0], old[0], rtol = rtol, atol = 0)) and  
            (np.isclose(new[1], old[1], rtol = rtol, atol = 0)) and 
            (np.isclose(new[2][0], old[2][0], rtol = rtol, atol = 0)) and 
            (np.isclose(new[2][1], old[2][1], rtol = rtol, atol = 0)) and 
            (np.isclose(new[3][0], old[3][0], rtol = rtol, atol = 0)) and 
            (np.isclose(new[3][1], old[3][1], rtol = rtol, atol = 0)) and 
            (np.isclose(new[4], old[4], rtol = rtol, atol = 0)) and 
            (np.isclose(new[5], old[5], rtol = rtol, atol = 0))):
            break
    return new
