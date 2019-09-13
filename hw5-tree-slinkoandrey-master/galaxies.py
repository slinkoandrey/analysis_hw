import numpy as np
import pandas
import matplotlib.pyplot as plt
import sklearn.model_selection as mod
import forest
import json

def main():
    data = pandas.read_csv('sdss_redshift.csv')
    x_train, x_test, y_train, y_test = mod.train_test_split(np.stack((data.u, data.g, data.r, data.i, data.z, data.r - data.i, data.g - data.z), axis = 1),
                                                                     np.array(data.redshift), test_size = 0.1)
    n = 15          #количество деревьев
    max_depth = 9   #максимальная глубина
    trees = forest.grow_forest(x_train, y_train, n, max_depth)
    y_pred = forest.predict(trees, x_test)
    plt.plot(y_test, y_pred, 'v')
    plt.plot(plt.xlim(), plt.xlim(), 'k', lw =0.5)
    plt.savefig('redhift.png')
    js = {"Cтандартное отклонение для всех подвыборок": np.std(y_test-y_pred)}
    with open('redhsift.json', 'w') as f:
        json.dump(js, f)
        
    data = pandas.read_csv('sdss.csv')
    x_new = np.stack((data.u, data.g, data.r, data.i, data.z, data.r - data.i, data.g - data.z), axis = 1)
    y_new = forest.predict(trees, x_new)
    redshift = pandas.Series(y_new)
    data = data.assign(redshift = redshift.values)
    data.to_csv('redshifts.csv')
    
    
if __name__ == '__main__':
    main()