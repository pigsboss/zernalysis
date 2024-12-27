#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

# initialize indices
n_max = 1000
idx = []
j_max = 0

def initidx():
    global idx, j_max
    for n in range(n_max+1):
        for m in range(-n, n+1, 2):
            idx.append((n, m))
    j_max = len(idx)+1

if len(idx)==0:
    initidx()

def zerndec(P, rho, phi, N=n_max):
    clst = []
    p = P
    j = 0
    for n in range(N+1):
        # print("decomposing n={}...".format(n))
        X = np.empty((p.size, n+1))
        for m in range(n+1):
            X[:,m] = zernval(j, rho, phi).ravel()
            j += 1
        c, res, _, _ = np.linalg.lstsq(X, p.ravel())
        # print("residual: {}".format(res))
        p = p - np.matmul(X, c)
        clst.append(c)
    return clst, p

def zernval(j, rho, phi):
    n, m = idx[j]
    R = np.zeros_like(rho)
    for k in range(0, (n-abs(m))//2+1):
        R += np.float64((-1)**k) * \
            np.float64(binom(n-k, k)) * \
            np.float64(binom(n-2*k, (n-abs(m))//2-k)) * \
            rho**(n-2*k)
    if m > 0:
        return R*np.cos(abs(m)*phi)
    elif m < 0:
        return R*np.sin(abs(m)*phi)
    else:
        return R
