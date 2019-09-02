import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# import multiprocessing
# multiprocessing.cpu_count()

def load_ccle(feature_type = 'expression', drug_target=None, normalize=False):
    if feature_type in ['expression', 'both']:
        # Load gene expression
        expression = pd.read_csv('data/expression.txt', delimiter='\t',
                                 header=2, index_col=1).iloc[:,1:]
        expression.columns = [c.split(' (ACH')[0] for c in expression.columns]
        features = expression
    if feature_type in ['mutation', 'both']:
        # Load gene mutation
        mutations = pd.read_csv('data/mutation.txt', delimiter='\t',
                                header=2, index_col=1).iloc[:,1:]
        mutations = mutations.iloc[[c.endswith('_MUT') for c in mutations.index]]
        features = mutations
    if feature_type == 'both':
        both_cells = set(expression.columns) & set(mutations.columns)
        z = {}
        for c in both_cells:
            exp = expression[c].values
            if len(exp.shape) > 1:
                exp = exp[:,0]
            z[c] = np.concatenate([exp, mutations[c].values])
        both_df = pd.DataFrame(z, index=[c for c in expression.index] +
                               [c for c in mutations.index])
        features = both_df
    response = pd.read_csv('data/response.csv', header=0, index_col=[0,2])
    # Get per-drug X and y regression targets
    cells = response.index.levels[0]
    drugs = response.index.levels[1]
    X_drugs = [[] for _ in drugs]
    y_drugs = [[] for _ in drugs]
    for j,drug in enumerate(drugs):
        if drug_target is not None and drug != drug_target:
            continue
        for i,cell in enumerate(cells):
            if cell not in features.columns or (cell, drug) not in response.index:
                continue
            X_drugs[j].append(features[cell].values)
            y_drugs[j].append(response.loc[(cell,drug), 'Amax'])
        print('{}: {}'.format(drug, len(y_drugs[j])))
    X_drugs = [np.array(x_i) for x_i in X_drugs]
    y_drugs = [np.array(y_i) for y_i in y_drugs]
    if normalize:
        X_drugs = [(x_i if (len(x_i) == 0) else (x_i - x_i.mean(axis=0,keepdims=True)) / x_i.std(axis=0).clip(1e-6)) for x_i in X_drugs]
        y_drugs = [(y_i if (len(y_i) == 0 or y_i.std() == 0) else (y_i - y_i.mean()) / y_i.std()) for y_i in y_drugs]
    return X_drugs, y_drugs, drugs, cells, features.index

X, y, drugs, cells, index = load_ccle()

X_train, X_test, y_train, y_test = train_test_split(X[1], y[1], test_size=0.2,  random_state=42)

np.savetxt('data/X_train.dat', X_train)
np.savetxt('data/y_train.dat', y_train)
np.savetxt('data/X_test.dat', X_test)
np.savetxt('data/y_test.dat', y_test)
