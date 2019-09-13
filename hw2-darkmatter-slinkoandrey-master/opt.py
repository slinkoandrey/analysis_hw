from collections import namedtuple
import numpy as np


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов можельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива меньше nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""

def gauss_newton(y, f , j, x0, k=1, tol=1e-4):
    """Метод Гаусса - Ньютона
    y — это массив с измерениями
    f(*x) — это функция от неивестных параметров, возвращающая значения, 
        рассчитанные в соответствии с моделью, в виде одномерного массива размера y.size
    j(*x) — это функция от неизвестных параметров, возвращающая якобиан 
        в виде двумерного массива (y.size, x0.size)
    x0 — массив с начальными приближениями параметров
    k — положительное число меньше единицы, параметр метода
    tol — условие сходимости, параметр метода
    Функция возвращает объект класса Result
    """
    x = np.asarray(x0, float)
    nfev = 0
    cost = []
    while True:
        r = y - f(*x)
        cost.append(0.5 * np.dot(r, r))
        nfev += 1
        jac = j(*x)
        grad = np.dot(jac.T, r)
        delta_x = np.linalg.solve(np.dot(jac.T, jac), grad)
        x += k*delta_x
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break
    return Result(nfev, cost, np.linalg.norm(grad), x)

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):  
    """Метод Левенберга—Марквардта
    y - массив c измерениями
    f(*x) — функция от неивестных параметров, возвращающая значения, 
        рассчитанные в соответствии с моделью,
        в виде одномерного массива размера y.size
    j(*x) — функция от неизвестных параметров, 
        возвращающая якобиан в виде двумерного массива (y.size, x0.size)
    x0 — массив с начальными приближениями параметров
    lmbd0 — начальное значение парметра lambda метода
    nu — мультипликатор для параметра lambda
    tol —  условие сходимости, параметр метода
    Функция возвращает объект класса Result
    """
    x = np.asarray(x0, float)
    nfev = 0
    cost = []
    c_old = np.inf
    while True:
        r = y - f(*x)
        jac = j(*x)
        grad = np.dot(jac.T, r)
        lmbd = lmbd0 * np.eye(*np.dot(jac.T, jac).shape)
        lmbd_nu = (lmbd0 / nu) * np.eye(*np.dot(jac.T, jac).shape)
        delta = np.linalg.solve(np.dot(jac.T, jac) + lmbd, grad)
        delta_nu = np.linalg.solve(np.dot(jac.T, jac) + lmbd_nu, grad)
        x += delta 
        x_nu = delta_nu + x
        r = y - f(*x)
        r_nu = y - f(*x_nu)
        c = 0.5 * np.dot(r, r)
        c_nu = 0.5 * np.dot(r_nu, r_nu)
        if c_nu <= c_old:
            x = x_nu
            r = r_nu
            c_old = c_nu
            lmbd0 *= 1/nu
        elif c_nu > c_old and c_nu <= c:
            c_old = c
        else:
            while c > c_old:
                lmbd0 *= nu
                lmbd = lmbd0 * np.eye(*np.dot(jac.T, jac).shape)
                delta = np.linalg.solve(np.dot(jac.T, jac) + lmbd, grad)
                x = delta + x
                r = y - f(*x)
                c = 0.5 * np.dot(r, r)
                nfev += 1
            c_old = c
        cost.append(c_old)
        nfev += 2
        if np.linalg.norm(delta_nu) <= tol * np.linalg.norm(x):
                break
    return Result(nfev, cost, np.linalg.norm(grad), x)

def conj_grad(A, c, x0, delta=np.inf):
    """Метод сопряженных градиентов
    A и c описывают квадратичную форму Q(x) = A^T A x + c
    x0 — начальное приближение
    delta задаёт размер области, в которой происходит поиск минимума
    """
    x = np.asarray(x0, dtype = np.float)
    n = x.shape
    n *= 2
    g = np.dot(A, x) + c
    p = -1 * g
    for i in range(n):
        a = np.dot(g, g) / np.dot(p, np.dot(A, p))
        x = x + a * p
        g_new = g + a * np.dot(A, p)
        b = np.dot(g_new, g_new) / np.dot(g, g)
        p = -1 * g_new + b * p
        g = g_new
        if np.abs(x - x0) > delta:
            break
    return x
        
