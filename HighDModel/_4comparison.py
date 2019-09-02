import numpy as np
import pandas as pd
import os
import pickle

def compare():
    count = 0
    for f in os.listdir('prediction accuracy'):
        if f.endswith(".dat"):
            count = count + 1

    results = pd.DataFrame({'method': ["" for x in range(count)], 'RMSE': np.zeros(count)})  

    i = 0
    for f in os.listdir('prediction accuracy'):
        if f.endswith(".dat"):
                results.loc[i,:] = [np.loadtxt('prediction accuracy/' + f), f]
                i = i + 1

    results.to_csv('prediction accuracy/results.csv', sep=',', index=False, header=True)
