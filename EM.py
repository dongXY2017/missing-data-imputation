# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:35:25 2018
对缺失数据进行EM填补（按照类别）
@author: lizzy
"""

import numpy as np
import impyute as impy
from delete import rand_delete
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

def get_mask(data):
    #m=0代表已观测值，m = 1表示缺失
    m = 1*np.isnan(data)
    return m

def em(data_un, loops):
    data = data_un.copy()
    label = np.unique(data[:,-1])
    t = label.shape[0]
    for k in range(t):
        null_xy = np.argwhere(np.isnan(data))
        for x_i, y_i in null_xy:
            col = data[data[:,-1]==label[k]][:,y_i]#按照类别找出
            null_catxy = np.argwhere(np.isnan(col))
            for x_cat_i in null_catxy[:,0]:
                mu = col[~np.isnan(col)].mean()
                std = col[~np.isnan(col)].std()
                col[x_cat_i] = np.random.normal(loc = mu, scale = std)
                previous, i = 1, 1
                for i in range(20):
                    #Expectation
                    mu = col[~np.isnan(col)].mean()
                    std = col[~np.isnan(col)].std()
                    #Maxmum
                    col[x_cat_i] = np.random.normal(loc = mu, scale = std)
                    dealt = (col[x_cat_i] - previous)/previous
                    if dealt < 0.1:
                        data[x_i, y_i] = col[x_cat_i]
                        break
                    data[x_i, y_i] = col[x_cat_i]
                    previous = col[x_cat_i]
    return data

if __name__ == '__main__':

    data_comp = np.loadtxt("spam.txt", delimiter = ",")#读入数据并对其进行随机删除
    data_uncomp = rand_delete(data_comp, 0.05)#对读入的数据进行删除
    m = get_mask(data_uncomp)#1代表缺失，0代表完整
#    data_comp1 = StandardScaler().fit_transform(data_comp)
#    data_comp2 = MinMaxScaler().fit_transform(data_comp)
    data_comp = Normalizer().fit_transform(data_comp)
    data_imp = em(data_uncomp, 50)#用自己编写的em算法进行填补
    data_impy = impy.em(data_uncomp)#用impyute中的em进行填补
    m_data_imp = data_imp*m
    m_data_raw = data_comp * m
    m_data_impy = data_impy * m#为了只计算填补的值之间的差距，故与m相乘
    error_impy = np.mean(np.square(m_data_raw - m_data_impy))
    print("error_impy:",error_impy)
    error = np.mean(np.square(m_data_raw - m_data_imp))
    print("define em error:", error)   
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    