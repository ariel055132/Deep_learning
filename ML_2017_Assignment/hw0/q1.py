#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:53:18 2020
hw0 in 2017 NTU-ML Assignment
@author: cindy
"""

import numpy as np

matrixA = []
for i in open("matrixA.txt"):
    row = [int(x) for x in i.split(",")]
    matrixA.append(row)
print(matrixA)
print(np.shape(matrixA))
print("======================")
matrixB = []
for i in open("matrixB.txt"):
    row = [int(x) for x in i.split(",")]
    matrixB.append(row)
print(matrixB)
print(np.shape(matrixB))

matrixA = np.array(matrixA)
matrixB = np.array(matrixB)
# shapes (50,1) and (500,1) not aligned: 1 (dim 1) != 500 (dim 0)
ans = matrixA.dot(matrixB)
ans.sort()
print(ans)
print(np.shape(ans))

# fml = '%i' save as int,  not float64 default by numpy
# delimiter : (str, optional)  String or character separating columns. (use endl to do separation)
np.savetxt("ans_one.txt", ans, fmt='%i', delimiter='\n')