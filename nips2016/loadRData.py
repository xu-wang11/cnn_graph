#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script Name : loadRData.py
# Author	  : Xu Wang(wangxu.93@hotmail.com)
# Date		  : 七月 11,2017
# Description : This file is for ...
import numpy as np
import rpy2.robjects as robjects
# import pandas.rpy.common as com
## load .RData and converts to pd.DataFrame
robjects.r.load('../data/beijing_clus.Rdata')
from scipy.io import savemat
def rdata_to_python(orig_data):
    data_type = type(orig_data)
    vector_type = [robjects.vectors.IntVector, robjects.vectors.FloatVector, robjects.vectors.FactorVector, robjects.vectors.Matrix, robjects.vectors.StrVector]
    if data_type in vector_type:
        return np.array(orig_data)
    elif data_type == robjects.vectors.ListVector or data_type == robjects.vectors.DataFrame:
        dict = {}
        for i in range(len(orig_data)):
            dict[orig_data.names[i]] = rdata_to_python(orig_data[i])
        return dict
    else:
        print(data_type)
        return 0

dest_data = rdata_to_python(robjects.r['alldata'])
savemat('rdata.mat', {'alldata': dest_data})
print("finish")
