# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:31:00 2021

@author: norah
"""

import numpy as np
from itertools import chain, combinations
from sys import maxsize
import time

def TSP(matrix):
    n = len(matrix)
    total = {}
    aps = set(i+1 for i in range(n))
    s = list(aps-{1})
    p = []
    d = {}
    p = {}
    for i, b in enumerate(chain.from_iterable(combinations(s, r) for r in range(len(s)+1))):
        total[b] = i
    ite = 0 
    path = []
    for i in range(0,n):
        d[(i,0)] = matrix[i][0]
    for k in range(1,n-1):
        for a,A in enumerate(combinations(aps-{1},k)):
            A = set(A)
            ap = aps-{1}-A
            A = list(A)
            A.sort()
            ty_list = []
            for j in A:
                ty = list(set(A) - {j})
                ty.sort()
                ty_list.append(ty)
            itt = total[tuple(A)]
            for i in ap:
                current = maxsize
                intt = 0
                for j in A:
                    ty = ty_list[intt]
                    ty = tuple(ty)
                    current_d = min(matrix[i-1][j-1]+d[(j-1,total[ty])], current)
                    d[(i-1,itt)] = current_d
                    if current_d == matrix[i-1][j-1]+d[(j-1,total[ty])]:
                        p[(i-1,itt)] = j
                    intt  = intt +1
            ite = ite +1
    glob = maxsize
    for j in range(2,n+1):
        indx = tuple(aps-{1,j})
        indx2 = tuple(aps-{1})
        glob = min(matrix[0][j-1]+d[(j-1,total[indx])],glob)
        if glob == matrix[0][j-1]+d[(j-1,total[indx])]:
            p[(0,total[indx2])] = j
    
    path.append(p[(0,total[indx2])])
    a = p[(0,total[indx2])]
    aps_copy = s
    for i in range(1,n-1):
        aps_copy.remove(a)
        a = p[(a-1,total[tuple(aps_copy)])]
        path.append(a)
    path.append(1)
    path.insert(0,1)
    d[(0,total[indx2])] = glob
    return path,d[(0,total[indx2])]              


input_size = 4
X = np.random.randint(1,31,size=(input_size,input_size))
X = np.triu(X,1)
X += X.T - np.diag(X.diagonal())
#X = list(X)

'''
X =[[0,11,1,1,3,20,21,22,13,13,18,1,19,10,14,11,14,5,2,21]
,[11,0,1,24,14,11,20,23,12,10,3,18,26,13,4,1,24,17,2,29]
,[1,1,0,14,27,20,26,28,9,28,14,3,24,7,23,8,10,8,5,20]
,[1,24,14,0,27,7,4,9,29,26,21,2,28,5,28,24,21,15,25,20]
,[3,14,27,27,0,6,11,18,14,21,5,6,5,12,10,5,28,14,13,24]
,[20,11,20,7,6,0,25,21,25,2,7,6,10,28,6,13,23,26,26,17]
,[21,20,26,4,11,25,0,3,27,20,14,12,23,24,7,19,4,23,26,4]
,[22,23,28,9,18,21,3,0,28,10,22,2,25,15,10,21,22,17,7,2]
,[13,12,9,29,14,25,27,28,0,6,23,10,26,8,28,14,5,18,1,28]
,[13,10,28,26,21,2,20,10,6,0,1,23,19,18,10,11,16,12,15,10]
,[18,3,14,21,5,7,14,22,23,1,0,13,21,10,11,3,11,25,11,23]
,[1,18,3,2,6,6,12,2,10,23,13,0,6,17,23,16,4,9,2,28]
,[19,26,24,28,5,10,23,25,26,19,21,6,0,1,22,17,27,3,23,1]
,[10,13,7,5,12,28,24,15,8,18,10,17,1,0,24,15,7,11,17,5]
,[14,4,23,28,10,6,7,10,28,10,11,23,22,24,0,6,4,2,4,8]
,[11,1,8,24,5,13,19,21,14,11,3,16,17,15,6,0,4,19,18,11]
,[14,24,10,21,28,23,4,22,5,16,11,4,27,7,4,4,0,21,12,26]
,[5,17,8,15,14,26,23,17,18,12,25,9,3,11,2,19,21,0,1,17]
,[2,2,5,25,13,26,26,7,1,15,11,2,23,17,4,18,12,1,0,24]
,[21,29,20,20,24,17,4,2,28,10,23,28,1,5,8,11,26,17,24,0]]
'''
t = time.time()
pat, cost_e = TSP(X)
runtime = round(time.time() - t, 3)
print("path:",pat)
print("cost:",cost_e)
print(f"Found optimal path in {runtime} seconds.") 
