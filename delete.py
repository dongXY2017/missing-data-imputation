# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:01:23 2018

@author: lizzy
"""
import numpy as np

def rand_delete(data, ratio):
    """
    input: complete data, missing ratio
    output: uncomplete data with random deletion under missing ratio(MCAR)
    """
    data_0 = data.copy()
    row, col = data_0.shape
    num = int(row*col*ratio)
    for i in range(num):
        data_0[np.random.randint(row), np.random.randint(col-1)] = np.nan
        
    return data_0
