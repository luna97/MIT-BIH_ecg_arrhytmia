# Copyright (c) Oezguen Turgut.
# All rights reserved.

# References: 
# Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI
# https://arxiv.org/abs/2308.05764

import os

import torch
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam=1e8, p=1e-2, niter=10):
    """
    Paul H. C. Eilers and Hans F.M. Boelens: Baseline Correction with Asymmetric Least Squares Smoothing
    """
    L = len(y)
    D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z
