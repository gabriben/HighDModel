#  from
# https://github.com/AlliedToasters/dfencoder/blob/master/dfencoder/dataframe.py

import numpy as np
import pandas as pd
import random

def swap(x, rs, likelihood=.15):

    """
    Performs random swapping of data.
    Each value has a likelihood of *argument likelihood*
        of being randomly replaced with a value from a different
        row.
    Returns a copy of the dataframe with equal size.
    """

    #select values to swap
    tot_rows = x.shape[0]
    n_rows = int(round(tot_rows*likelihood))
    n_cols = x.shape[1]
    print n_cols


    column = np.repeat(np.arange(n_cols).reshape(1, -1),
                       repeats=n_rows, axis=0)
    random.seed(rs)
    row = np.random.randint(0, tot_rows, size=(n_rows, n_cols))

    # row, column = gen_indices()
    new_mat = x.values
    to_place = new_mat[row, column]

    column = np.repeat(np.arange(n_cols).reshape(1, -1),
                       repeats=n_rows, axis=0)
    random.seed(rs+1)
    row = np.random.randint(0, tot_rows, size=(n_rows, n_cols))

    # row, column = gen_indices()
    new_mat[row, column] = to_place

    # dtypes = {col:typ for col, typ in zip(x.columns, x.dtypes)}
    # result = EncoderDataFrame(columns=x.columns, data=new_mat)
    # result = result.astype(dtypes, copy=False)

    return result
