import numpy as np
import pandas as pd
import os
import pickle

def modelling(X_train, y_train, X_test, y_test, fs = 'lasso', method = 'ols',
              select = 500):
    
    if method == 'ols':
        from sklearn.linear_model import LinearRegression
        mod = LinearRegression().fit(X_train, y_train)

    if method == 'elasticNet':
        from sklearn.linear_model import ElasticNetCV
        mod = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                        cv=10, tol = 0.001, n_jobs = 7)
        mod.fit(X_train,y_train)

    if method == 'xgboost':
        import xgboost as xg
        max_depth = 3
        min_child_weight = 10
        subsample = 0.5
        colsample_bytree = 0.6
        objective = 'reg:linear'
        num_estimators = 1000
        learning_rate = 0.3

        mod = xg.XGBRegressor(max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        objective=objective,
                        n_estimators=num_estimators,
                        learning_rate=learning_rate)
        mod.fit(X_train,y_train)
        
        # implement CV
        
    if method == 'nn':
        from sklearn.preprocessing import StandardScaler
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.callbacks import EarlyStopping
        from keras.callbacks import ModelCheckpoint
        from keras.models import load_model

        mod = Sequential()
        # input layer
        mod.add(Dense(50, activation='relu', input_shape=(int(select),)))
        # hidden layer
        mod.add(Dense(50, activation='relu'))
        # output layer 
        mod.add(Dense(1, activation='linear'))
        # mod.summary()
        mod.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

        # patient early stopping and select best model (not always the last)
        es = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=200)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc',
                             mode='max', verbose=1, save_best_only=True)

        history = mod.fit(X_train, y_train, epochs=1000, 
                          batch_size=25, verbose=1,
                            validation_data = (Xt, y_test),
                             callbacks=[es])
        
    # pickle.dump(mod, open('models/' + fs + '_' + method + '_' +
    #                        select + '.sav', 'wb'))

    if method == 'nn':
        rmse = (sum((np.concatenate(mod.predict(X_test)) - y_test)**2) /
              y_test.size)**(.5)
    else:
        rmse = (sum((mod.predict(X_test) - y_test)**2) / y_test.size)**(.5)

    return mod, rmse
        
    # np.savetxt('prediction accuracy/' + fs + '_' + method + '_' +
    #                select + '.dat', np.array([rmse]))

