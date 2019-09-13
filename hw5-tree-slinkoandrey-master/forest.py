import numpy as np
import tree

def grow_forest(x, y, n, max_depth):
    forest = np.empty(n, dtype=object)
    for i in range(n):
        indexes = np.random.random_integers((x.shape[0]-1), size=((x.shape[0]) // n))
        xi = x[indexes][:]
        yi = y[indexes]
        forest[i] = tree.build_tree(xi, yi, max_depth=max_depth)
    return forest

def predict(forest, x):
    y_i = np.empty(forest.size, dtype=object)
    for i in range(forest.size):
        y_i[i] = tree.predict(forest[i], x) 
    return np.mean(y_i)