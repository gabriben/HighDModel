import numpy as np
import pandas as pd
import h5py

import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model


# X = np.loadtxt('data/X_train.dat')
# y = np.loadtxt('data/y_train.dat')

# continuous response feature selection
def featureSelection(X, y, method = 'lasso', select = 500):
    
    t0 = time.time()
    
    # sparse (15 seconds)
    if method == 'lasso':
        from sklearn import linear_model
        
        a = 0.861 if select == 500 else 0.0755
        lasso = linear_model.Lasso(alpha = a)
        lasso.fit(X,y)
        XSelected = X[:,lasso.coef_ != 0]
        indices = np.where(lasso.coef_ != 0)
        if indices > select:
            indices = np.argsort(-lasso.coef_)[:select]
    
    # non-sparse (157 seconds)
    if method == 'rf':
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        
        t = ExtraTreesRegressor(n_estimators=50)
        t.fit(X, y)
        model = SelectFromModel(t, prefit=True,
                                max_features = select)
        XSelected = model.transform(X)
        indices = np.where(model.get_support)
    
    # non-sparse (8.5 seconds)
    if method == 'svm':
        from sklearn.svm import SVR
        from sklearn.feature_selection import SelectFromModel
        
        SVMReg = SVR(kernel = 'linear',
                     gamma='scale', C=1.0, epsilon=0.2)
        SVMReg.fit(X, y)
        model = SelectFromModel(SVMReg, prefit=True, 
                                max_features = select)
        XSelected = model.transform(X)
        indices = np.where(model.get_support())
    
    # wrapper model (preset number of features) (1000 seconds / 5000 seconds)
    if method == 'hsiclasso':
        from pyHSICLasso import HSICLasso
        
        hsic_lasso = HSICLasso()
        hsic_lasso.input(X,y)
        hsic_lasso.regression(select)
        XSelected = X[:,hsic_lasso.get_index()]
        indices = hsic_lasso.get_index()

    # dimensionality reduction
        # PCA
        # MDS
        # PLS
        # DWT
        
#    f = h5py.File('selected/' + str(select) + '/X_' + method + '.hdf5', "w")
#    f.create_dataset('X', data=XSelected)
#    f.create_dataset('indices', data=indices)
#    f.close()

    # return indices
    np.savetxt('selected/' + str(select) + '/X_' + method + '.dat', indices)
    
    # np.savetxt('selected/' + str(select) + '/X_' + method + '.dat', XSelected)

    print("--- %s seconds ---" % (time.time() - t0))

    
    
# methods = ['lasso'] # 'lasso', 'rf', 'svm'
# for m in methods:
#     for s in [500,1000]:
#         featureSelection(X, y, m, s)
    
    
# i.size
