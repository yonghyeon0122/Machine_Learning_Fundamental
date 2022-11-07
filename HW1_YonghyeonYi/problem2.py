# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:57:41 2021

@author: yi000055
"""

import numpy as np

A = [[1,1,1,1],
     [1,2,4,8],
     [1,3,9,27],
     [1,4,16,64]]
At = np.transpose(A)

#problem2-1
print("===problem2-1====")
print(np.trace(A))
print(np.trace(At))
print(np.trace(np.matmul(At, A)))
print(np.trace(np.matmul(A,At)))

#problem2-3
print("===problem2-3====")
print(np.linalg.matrix_rank(A))
