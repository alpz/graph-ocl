import torch
from torch.autograd import Function

#import osqp
import numpy as np
import scipy.sparse as sp
import torch.sparse

import scipy.sparse.linalg as spla
import cupy as cp
import cupyx.scipy.sparse as cps
import cupyx.scipy.sparse.linalg as cpsla


def solve_grad(gamma, g, A, AAt_LU):
    _g = g.T

    Ag = (A@_g).T

    r = lu_solve(AAt_LU, Ag)
    # r is (b, n)
    t = (A.T @ r.T).T

    dx = 1/gamma*(g - t)

    dx = -dx

    return dx


def solve_forward(gamma, p, Eb, A, AAt_LU):
    _p = p.T

    #print(A.shape, _p.shape)
    Ap = (A@_p).T

    r = lu_solve(AAt_LU, Ap)
    # r is (b, n)
    t = (A.T @ r.T).T

    x = 1/gamma*(p - t)
    x = x + Eb
    x = -x

    return x


def lu_solve(A_LU, b):
    """b in a batch (b, n) """

    b = b.T
    Pr = A_LU[0]
    L = A_LU[1]
    U = A_LU[2]
    Pc = A_LU[3]

    b = Pr.T @ b
    #L y = b
    y = cpsla.spsolve_triangular(L, b, overwrite_b=True, lower=True).astype(np.float32)

    # UPc x = y
    Pc_x = cpsla.spsolve_triangular(U, y, overwrite_b=True, lower=False).astype(np.float32)

    x = Pc.T @ Pc_x

    # restore batch dim
    x = x.T

    return x



def QPFunctionEq(A_=None,b_=None,AAt_LU=None, gamma=0.1, DEVICE='cuda'):
    A=A_
    t = lu_solve(AAt_LU, -b_)
    Eb = A.T @ t

    class QPFunctionFn(Function):
        @staticmethod
        def forward(ctx, p_):
            p_ = cp.asarray(p_.detach())
            x= solve_forward(gamma, p_, Eb, A, AAt_LU)
            x = torch.as_tensor(x, device=DEVICE)
            zhats = x

            return zhats

        @staticmethod
        def backward(ctx, dl_dzhat):
            dl_dzhat = cp.asarray(dl_dzhat.detach())
            dx = solve_grad(gamma, dl_dzhat, A, AAt_LU)
            dx = torch.as_tensor(dx, device=DEVICE)
            return dx

    return QPFunctionFn.apply


