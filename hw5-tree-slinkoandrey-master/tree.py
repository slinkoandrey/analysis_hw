import numpy as np
from collections import namedtuple
from scipy.optimize import minimize_scalar

Node = namedtuple ('Node', ('feature', 'value', 'impurity', 'left', 'right'))
Leaf = namedtuple ('Leaf', ('value', 'x', 'y'))

def criteria(y):
    return np.std(y) ** 2

def impurity(y_left, y_right):
    size = y_left.size + y_right.size
    h = (y_left.size * criteria(y_left) + y_right.size * criteria(y_right)) / size
    return h

def partition(x, y, feature, value):
    i_right = x[:, feature] >= value
    i_left = np.logical_not(i_right)
    return (x[i_left], y[i_left]), (x[i_right], y[i_right])

def f(value, feature, x, y):
    (_, y_left), (_, y_right) = partition(x, y, feature, value)
    return impurity(y_left, y_right) 

def find_best_split(x, y):   
    best_feature, best_value, best_impurity = 0, x[0,0], np.inf
    for feature in range(x.shape[1]):
        x_i_sorted = np.sort(x[:, feature])
        result = minimize_scalar(
                f,
                args = (feature, x, y),
                method = 'Bounded',
                bounds = (x_i_sorted[1], x_i_sorted[-1]))
        assert result.success
        value = result.x
        impurity = result.fun
        if impurity < best_impurity:
            best_feature, best_value, best_impurity = feature, value, impurity
    return best_feature, best_value, best_impurity

def build_tree(x, y, depth = 1, max_depth = np.inf):
    if depth == max_depth or criteria(y) < 1e-4:
        return Leaf(np.mean(y), x, y)
    feature, value, impurity = find_best_split(x, y)
    (x_left, y_left), (x_right, y_right) = partition(x, y, feature, value)
    left = build_tree(x_left, y_left, depth+1, max_depth)
    right = build_tree(x_right, y_right, depth+1, max_depth)
    root = Node(feature, value, impurity, left, right)
    return root

def predict(tree, x):
    y = np.empty(x.shape[0])
    for i, row in enumerate(x):
        node = tree
        while isinstance(node, Node):
            if row[node.feature] >= node.value:
                node = node.right
            else:
                node = node.left
        y[i] = node.value
    return y