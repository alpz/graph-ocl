import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cps
import time
import torch
from sksparse.cholmod import cholesky, cholesky_AAt

from cut.qp import QPFunctionEq#,QPSolvers

class STCut():
    def __init__(self, N=64, K=1, gamma=0.5, cuda=True):
        self.N = 64
        #not used
        self.K = K
        self.gamma = gamma

        self.DEVICE = torch.device("cuda" if cuda else "cpu")

    def get_edge_index(self, xi, xj, yi, yj, c_idx):
        # c_idx component index 0..K-1
        # ij st i:1 to N corresponds to image
        # i=0 and N+1 correspond to padding
        N = self.N
        K = self.K
        edge_offset = 2 + (N) * (N) * K
        C = N * N

        if xi == 's':
            return (edge_offset + C * c_idx + yi * (N) + yj)  # - 1
        elif xi == 't':
            return (edge_offset + C * K + C * c_idx + yi * (N) + yj)  # - 1
        else:
            # return (edge_offset + 2 * (N + 2) * (N + 2) + yi * 4 * (N + 2) + yj * 4) - 1
            kchannel_offset = (N) * (N)*4*c_idx
            if yi == xi and yj == xj - 1:
                f = 3
            elif yi == xi + 1 and yj == xj:
                f = 0
            elif yi == xi and yj == xj + 1:
                f = 1
            elif yi == xi - 1 and yj == xj:
                f = 2
            # for downward diag (not used)
            elif yi == xi + 1 and yj == xj + 1:
                f = 2
            else:
                raise Exception('error')

            # return (edge_offset + 2 * (N + 2) * (N + 2) + yi * 4 * (N + 2) + yj * 4) - 1
            return (edge_offset + 2 * (N) * (N) * K + kchannel_offset+ f*N*N + xi * N + xj)  # - 1


    def get_node_index(self, xi, xj, c_idx):
        N = self.N
        return 2 + N * N * c_idx + xi * (N) + xj


    def make_constraints(self):
        def slack_index(constraint_idx):
            n_num_vars = 2 + 7 * (N) * (N) * K
            return n_num_vars + constraint_idx

        N = self.N
        K = self.K
        # duv - pu + pv >= 0
        row_list = []
        col_list = []
        val_list = []
        b_list = []
        row = 0
        for c_idx in range(K):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    ij = self.get_node_index(i, j, c_idx)
                    idx1 = self.get_edge_index(i, j, i, j - 1, c_idx)
                    idx2 = self.get_edge_index(i, j, i + 1, j, c_idx)
                    idx3 = self.get_edge_index(i, j, i, j + 1, c_idx)
                    idx4 = self.get_edge_index(i, j, i - 1, j, c_idx)

                    val_list.extend([1, -1, 1, -1])
                    col_list.extend([idx1, ij, self.get_node_index(i, j - 1, c_idx), slack_index(row)])
                    row_list.extend([row, row, row, row])
                    row = row + 1

                    val_list.extend([1, -1, 1, -1])
                    col_list.extend([idx2, ij, self.get_node_index(i + 1, j, c_idx), slack_index(row)])
                    row_list.extend([row, row, row, row])
                    row = row + 1

                    val_list.extend([1, -1, 1, -1])
                    col_list.extend([idx3, ij, self.get_node_index(i, j + 1, c_idx), slack_index(row)])
                    row_list.extend([row, row, row, row])
                    row = row + 1

                    val_list.extend([1, -1, 1, -1])
                    col_list.extend([idx4, ij, self.get_node_index(i - 1, j, c_idx), slack_index(row)])
                    row_list.extend([row, row, row, row])
                    row = row + 1

                    b_list.extend([0, 0, 0, 0])

        # s-v edges
        for c_idx in range(K):
            for i in range(0, N):
                for j in range(0, N):
                    idx1 = self.get_edge_index('s', 's', i, j, c_idx)

                    val_list.extend([1, -1, 1, -1])
                    col_list.extend([idx1, 0, self.get_node_index(i, j, c_idx), slack_index(row)])
                    row_list.extend([row, row, row, row])
                    b_list.append(0)

                    row = row + 1

        # v-t edges
        for c_idx in range(K):
            for i in range(0, N):
                for j in range(0, N):
                    idx1 = self.get_edge_index('t', 't', i, j, c_idx)

                    val_list.extend([1, -1, 1, -1])
                    col_list.extend([idx1, self.get_node_index(i, j, c_idx), 1, slack_index(row)])
                    row_list.extend([row, row, row, row])
                    b_list.append(0)
                    row = row + 1



        #using dist of 10
        val_list.extend([1, -1])
        col_list.extend([0, 1])
        row_list.extend([row, row])
        b_list.append(10)
        row = row + 1

        return np.array(val_list), np.array(row_list), np.array(col_list), np.array(b_list), row


    def solve(self, c_params):
        c_params = c_params.reshape(c_params.shape[0],-1)
        zero_params = torch.zeros((c_params.shape[0], 2+(self.N)*(self.N)*self.K), device=self.DEVICE)
        slack_params = torch.zeros((c_params.shape[0], self.num_constraints), device=self.DEVICE)
        c_params = torch.cat([zero_params, c_params, slack_params], 1)

        x = self.qpf_q(c_params)

        x = x[:, :self.num_vars]
        return x

    def get_node_vars(self, x):
        #extract node variables
        x = x[:, 2:2+self.K*((self.N)**2)].reshape((-1,self.K,self.N,self.N))

        return x

    def get_edge_vars(self, x):
        #extract node variables
        offset = 2+ self.K*(self.N)**2
        q_offset = offset + 2*self.K*(self.N)**2
        size = self.K*(self.N)**2

        #s
        x = x[:, offset:offset+size].reshape((-1,self.K,self.N,self.N))
        return x

    def get_edge_vars_st(self, x):
        #extract node variables
        offset = 2+ self.K*(self.N)**2
        q_offset = offset + 2*self.K*(self.N)**2
        size = self.K*(self.N)**2

        s = x[:, offset:offset+size].reshape((-1,1,self.K,self.N,self.N))
        t = x[:, offset+size:offset+2*size].reshape((-1,1,self.K,self.N,self.N))

        st = torch.cat([s,t], dim=1)

        x = torch.softmax(st,dim=1)[:,0,:,:,:]
        return x


    #def get_constraints(self):
    def build_solver(self):
        num_vars = 2 + 7 * (self.N) * (self.N) * self.K
        val, row, col, rhs, num_constraints = self.make_constraints()

        self.num_vars = num_vars
        self.num_constraints = num_constraints

        # add slack vars
        A = sp.csr_matrix((val, (row.astype(np.intc), col.astype(np.intc))),
                          shape=(num_constraints, num_vars + num_constraints))
        lu_AAt = get_cholesky(A)
        A = cps.csr_matrix(A.astype(np.float32))
        rhs = cp.asarray(rhs.astype(np.float32))

        self.qpf_q = QPFunctionEq(A_=A, b_=rhs, AAt_LU=lu_AAt, DEVICE=self.DEVICE, gamma=self.gamma)

        return A, rhs, lu_AAt, num_vars, num_constraints, self.K


def get_cholesky(A):
    print('Begin Cholesky')
    start = time.time()

    F = cholesky_AAt(A, ordering_method='nesdis')

    L = F.L().astype(np.float32)
    P = F.P().astype(np.float32)

    P = P.astype(np.int32)

    r = L.shape[0]
    c = L.shape[1]

    P = sp.csr_matrix((np.ones(r), (P, np.arange(r))))

    print('End Cholesky ', time.time() - start)
    print('Density ', L.nnz / np.prod(L.shape))

    def sz(a):
        return a.data.nbytes + a.indptr.nbytes + a.indices.nbytes

    print("Size ", (sz(L) + sz(P)) / (1000 * 1000))

    Pr = cps.csr_matrix(P).astype(np.float32)
    Pc = Pr.T  # cps.csr_matrix(Pc).astype(np.float32)
    L = cps.csr_matrix(L).astype(np.float32)
    # setting U = L.T causes a memory leak
    U = L.T.tocsr()

    return (Pr, L, U, Pc)

