# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import copy
import warnings
import tensorly as tl
import scipy as sp
from scipy.linalg import hankel
from scipy import linalg
import numpy as np
import torch.nn as nn
from tensorly.decomposition import tucker


def svd_fun(matrix, n_eigenvecs=None):
    """Computes a fast partial SVD on `matrix`
    If `n_eigenvecs` is specified, sparse eigendecomposition is used on
    either matrix.dot(matrix.T) or matrix.T.dot(matrix).
    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors
    """

    # Choose what to do depending on the params
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
        max_dim = dim_2
    else:
        min_dim = dim_2
        max_dim = dim_1

    if n_eigenvecs >= min_dim:
        if n_eigenvecs > max_dim:
            warnings.warn(('Trying to compute SVD with n_eigenvecs={0}, which '
                           'is larger than max(matrix.shape)={1}. Setting '
                           'n_eigenvecs to {1}').format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim

        if n_eigenvecs is None or n_eigenvecs > min_dim:
            full_matrices = True
        else:
            full_matrices = False

        # Default on standard SVD
        U, S, V = sp.linalg.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = sp.sparse.linalg.eigsh(
                np.dot(matrix, matrix.T.conj()), k=n_eigenvecs, which='LM'
            )
            S = np.sqrt(S)
            V = np.dot(matrix.T.conj(), U * 1 / S[None, :])
        else:
            S, V = sp.sparse.linalg.eigsh(
                np.dot(matrix.T.conj(), matrix), k=n_eigenvecs, which='LM'
            )
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1 / S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        V = V.T.conj()
    return U, S, V

def svd_init(tensor, modes, ranks):
    factors = []
    for index, mode in enumerate(modes):
        eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=ranks[index])
        factors.append(eigenvecs)
        #print("factor mode: ", index)
    return factors

def autocorr(Y, lag=10):
    """
    计算<Y(t), Y(t-0)>, ..., <Y(t), Y(t-lag)>
    :param Y: list [tensor1, tensor2, ..., tensorT]
    :param lag: int
    :return: array(k+1)
    """
    T = len(Y)
    r = []
#     print("Y")
#     print(Y)
    for l in range(lag+1):
        product = 0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(Y[t] * Y[tl])
        r.append(product)
    return r

def fit_ar(Y, p=10):
    r = autocorr(Y, p)
    #print("auto-corr:",r)
    R = linalg.toeplitz(r[:p])
    r = r[1:]
    A = linalg.pinv(R).dot(r)
    return A

def fit_ar_ma(Y, p=10, q=1):
    # print("fit_ar_ma")
    N = len(Y)

    A = fit_ar(Y, p)
    B = [0.]
    if q > 0:
        Res = []
        for i in range(p, N):
            res = Y[i] - np.sum([a * Y[i - j] for a, j in zip(A, range(1, p + 1))], axis=0)
            Res.append(res)
        # Res = np.array(Res)
        B = fit_ar(Res, q)
    return A, B

def unfold(tensor,n):
    size = np.array(tensor.shape)
    N = size.shape[0]
    I = int(size[n])
    J = int(int(np.prod(size)) / int(I))
    pmt=np.array(range(n,n+1))
    pmt=np.append(pmt,range(0,n))
    pmt=np.append(pmt,range(n+1,N)).astype(np.int)
    return np.reshape(np.transpose(tensor, pmt),[I, J])

def fold(matrix,n,size_t_ori):
    N = np.array(size_t_ori).shape[0]
    size_t_pmt = np.concatenate([size_t_ori[n:(n+1)],size_t_ori[0:n],size_t_ori[(n+1):N]], axis=0)
    pmt = np.array(range(1,n+1))
    pmt = np.append(pmt,range(0,1))
    pmt = np.append(pmt,range(n+1,N)).astype(np.int)
    return np.transpose(np.reshape(matrix,size_t_pmt),pmt)

def make_duplication_matrix(T,tau):
    H = hankel(range(tau),range(tau-1,T))
    T2= np.prod(H.shape)
    h = np.reshape(H,[1,T2])
    h2= np.array([range(T2)])
    index = np.concatenate([h,h2],axis=0)
    S = np.zeros([T,T2], dtype='uint64')
    S[tuple(index)]=1
    return S.T

def tmult(tensor,matrix,n):
    size = np.array(tensor.shape)
    size[n] = matrix.shape[0]
    return fold(np.matmul(matrix,unfold(tensor,n)),n,size)

def hankel_tensor(x,TAU):
    N = len(TAU)
    N2= N*2
    T2  = np.zeros([N,2],dtype='uint64')
    S = list()
    Hx = x
    for n in range(N):
        tau   = TAU[n]
        T     = x.shape[n]
        T2[n,:] = [tau,T-tau+1]
        S.append(make_duplication_matrix(x.shape[n],TAU[n]))
        Hx = tmult(Hx,S[n],n)
    size_h_tensor = np.reshape(T2,[N2,])
    Hx = np.reshape(Hx,size_h_tensor)
    return Hx, S

def hankel_tensor_adjoint(Hx,S):
    N = len(S)
    size_h_tensor = np.zeros([N,], dtype='uint64')
    for n in range(N):
        size_h_tensor[n] = S[n].shape[0]
    Hx = np.reshape(Hx,size_h_tensor)
    for n in range(N):
        Hx = tmult(Hx,S[n].T,n)
    return Hx

class MDTWrapper(object):

    def __init__(self, data, tau=None):
        self._data = data.astype(np.float32)
        self._ori_data = data.astype(np.float32)
        self.set_tau(tau)
        is_transformed = False
        self._ori_shape = data.shape
        pass

    def set_tau(self, tau):
        if isinstance(tau, np.ndarray):
            self._tau = tau
        elif isinstance(tau, list):
            self._tau = np.array(tau)
        else:
            raise TypeError(" 'tau' need to be a list or numpy.ndarray")

    def get_tau(self):
        return self._tau

    def shape(self):
        return self._data.shape

    def get_data(self):
        return self._data

    def get_ori_data(self):
        return self._ori_data

    def transform(self, tau=None):
        _tau = tau if tau is not None else self._tau
        result, S = hankel_tensor(self._data, _tau)
        self.is_transformed = True
        # print("before squeeze: ", result.shape)
        axis_dim = tuple(i for i, dim in enumerate(result.shape) if dim == 1 and i != 0)
        result = np.squeeze(result, axis=axis_dim)
        # print("after squeeze: ", result.shape)
        self._data = result
        return result

    def inverse(self, data=None, tau=None, ori_shape=None):
        _tau = tau if tau is not None else self._tau
        _ori_shape = ori_shape if ori_shape is not None else self._ori_shape
        _data = data if data is not None else self._data
        O = np.ones(_ori_shape, dtype='uint8')
        Ho, S = hankel_tensor(O.astype(np.float32), _tau)
        D = hankel_tensor_adjoint(Ho, S)

        result = np.divide(hankel_tensor_adjoint(_data, S), D)
        self.is_transformed = False
        self._data = result
        return result

    def predict(self):
        '''
        # To do:
        # predict function
        '''
        pass

class BHTARIMA(object):

    def __init__(self, config, data_feature):
        self._ts_ori_shape = data_feature['ts_ori_shape']
        self._N = data_feature['N']
        self.T = data_feature['T']

        self.config = config
        self._p = self.config.get('p')
        self._d = self.config.get('d')
        self._q = self.config.get('q')
        self._taus = self.config.get('taus')
        self._Rs = self.config.get('Rs')
        self._K = self.config.get('K')
        self._tol = self.config.get('tol')
        self._Us_mode = self.config.get('Us_mode')
        self._verbose = self.config.get('verbose')
        self._convergence_loss = self.config.get('convergence_loss')

        # check Rs parameters
        M = 0
        for dms, tau in zip(self._ts_ori_shape, self._taus):
            # print(dms, ',', tau)
            if dms == tau:
                M += 1
            elif dms > tau:
                M += 2
        # M=3
        if M - 1 != len(self._Rs):
            raise ValueError("the first element of taus should be equal to the num of series")

    def _forward_MDT(self, data, taus):
        self.mdt = MDTWrapper(data, taus)
        trans_data = self.mdt.transform()
        self._T_hat = self.mdt.shape()[-1]
        return trans_data, self.mdt

    def _initilizer(self, T_hat, Js, Rs, Xs):

        # initilize Us
        U = [np.random.random([j, r]) for j, r in zip(list(Js), Rs)]

        # initilize es
        begin_idx = self._p + self._q
        es = [[np.random.random(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]

        return U, es

    def _test_initilizer(self, trans_data, Rs):

        T_hat = trans_data.shape[-1]
        # initilize Us
        U = [np.random.random([j, r]) for j, r in zip(list(trans_data.shape)[:-1], Rs)]

        # initilize es
        begin_idx = self._p + self._q
        es = [[np.zeros(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]
        return U, es

    def _initilize_U(self, T_hat, Xs, Rs):

        haveNan = True
        while haveNan:
            factors = svd_init(Xs[0], range(len(Xs[0].shape)), ranks=Rs)
            haveNan = np.any(np.isnan(factors))
        return factors

    def _inverse_MDT(self, mdt, data, taus, shape):
        return mdt.inverse(data, taus, shape)

    def _get_cores(self, Xs, Us):
        cores = [tl.tenalg.multi_mode_dot(x, [u.T for u in Us], modes=[i for i in range(len(Us))]) for x in Xs]
        return cores

    def _estimate_ar_ma(self, cores, p, q):
        cores = copy.deepcopy(cores)
        alpha, beta = fit_ar_ma(cores, p, q)

        return alpha, beta

    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor, list):
            return [tl.base.fold(ten, mode, shape) for ten in tensor]
        elif isinstance(tensor, np.ndarray):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _get_unfold_tensor(self, tensor, mode):

        if isinstance(tensor, list):
            return [tl.base.unfold(ten, mode) for ten in tensor]
        elif isinstance(tensor, np.ndarray):
            return tl.base.unfold(tensor, mode)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _update_Us(self, Us, Xs, unfold_cores, n):

        T_hat = len(Xs)
        M = len(Us)
        begin_idx = self._p + self._q

        H = self._get_H(Us, n)
        # orth in J3
        if self._Us_mode == 1:
            if n < M - 1:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
            else:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
        # orth in J1 J2
        elif self._Us_mode == 2:
            if n < M - 1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        # no orth
        elif self._Us_mode == 3:
            As = []
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            a = sp.linalg.pinv(np.sum(As, axis=0))
            b = np.sum(Bs, axis=0)
            temp = np.dot(a, b)
            Us[n] = temp / np.linalg.norm(temp)
        # all orth
        elif self._Us_mode == 4:
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            b = np.sum(Bs, axis=0)
            # b = b.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
            U_, _, V_ = np.linalg.svd(b, full_matrices=False)
            Us[n] = np.dot(U_, V_)
        # only orth in J1
        elif self._Us_mode == 5:
            if n == 0:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        # only orth in J2
        elif self._Us_mode == 6:
            if n == 1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        return Us

    def _update_Es(self, es, alpha, beta, unfold_cores, i, n):

        T_hat = len(unfold_cores)
        begin_idx = self._p + self._q

        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii + 1)] for ii in range(self._p)], axis=0)
            b = np.sum(
                [beta[j] * self._get_unfold_tensor(es[t - begin_idx][-(j + 1)], n) for j in range(self._q) if i != j],
                axis=0)
            As.append(unfold_cores[t] - a + b)
        E = np.sum(As, axis=0)
        for t in range(len(es)):
            es[t][i] = self._get_fold_tensor(E / (2 * (begin_idx - T_hat) * beta[i]), n, es[t][i].shape)
        return es

    def _compute_convergence(self, new_U, old_U):

        new_old = [n - o for n, o in zip(new_U, old_U)]

        a = np.sum([np.sqrt(tl.tenalg.inner(e, e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e, e)) for e in new_U], axis=0)
        return a / b

    def _tensor_difference(self, d, tensors, axis):
        """
        get d-order difference series

        Arg:
            d: int, order
            tensors: list of ndarray, tensor to be difference

        Return:
            begin_tensors: list, the first d elements, used for recovering original tensors
            d_tensors: ndarray, tensors after difference

        """
        d_tensors = tensors
        begin_tensors = []

        for _ in range(d):
            begin_tensors.append(d_tensors[0])
            d_tensors = list(np.diff(d_tensors, axis=axis))

        return begin_tensors, d_tensors

    def _tensor_reverse_diff(self, d, begin, tensors, axis):
        """
        recover original tensors from d-order difference tensors

        Arg:
            d: int, order
            begin: list, the first d elements
            tensors: list of ndarray, tensors after difference

        Return:
            re_tensors: ndarray, original tensors

        """

        re_tensors = tensors
        for i in range(1, d + 1):
            re_tensors = list(np.cumsum(np.insert(re_tensors, 0, begin[-i], axis=axis), axis=axis))

        return re_tensors

    def _update_cores(self, n, Us, Xs, es, cores, alpha, beta, lam=1):

        begin_idx = self._p + self._q
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Us, n)
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            b = np.sum([beta[i] * self._get_unfold_tensor(es[t - begin_idx][-(i + 1)], n) for i in range(self._q)],
                       axis=0)
            a = np.sum([alpha[i] * self._get_unfold_tensor(cores[t - (i + 1)], n) for i in range(self._p)], axis=0)
            unfold_cores[t] = 1 / (1 + lam) * (lam * np.dot(np.dot(Us[n].T, unfold_Xs), H.T) + a - b)
        return unfold_cores

    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [trans_data[..., t] for t in range(T_hat)]

        return Xs

    def _get_H(self, Us, n):

        Hs = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i != n])
        return Hs

    def run(self, data):
        """run the program

        Returns
        -------
        result : np.ndarray, shape (num of items, num of time step +1)
            prediction result, included the original series

        loss : list[float] if self.convergence_loss == True else None
            Convergence loss

        """
        self._ts = data
        result, loss = self._run()

        if self._convergence_loss:
            return result, loss

        return result, None

    def _run(self):

        # step 1a: MDT
        # transfer original tensor into MDT tensors
        trans_data, mdt = self._forward_MDT(self._ts, self._taus)
        # print('trans_data.shape: ', trans_data.shape) [228, 5, 35]
        Xs = self._get_Xs(trans_data)
        # print('len(Xs): ', len(Xs)) 35
        # print('Xs[0].shape: ', Xs[0].shape) [228, 5]
        if self._d != 0:
            begin, Xs = self._tensor_difference(self._d, Xs, 0)
        # print('len(begin): ', len(begin)) 2
        # print('begin[0].shape: ', begin[0].shape) [228, 5]
        # print('len(Xs): ', len(Xs)) 33
        # print('Xs[0].shape: ', Xs[0].shape) [228, 5]

        # for plotting the convergence loss figure
        con_loss = []

        # Step 2: Hankel Tensor ARMA based on Tucker-decomposition

        # initialize Us
        Us, es = self._initilizer(len(Xs), Xs[0].shape, self._Rs, Xs)

        for k in range(self._K):
            old_Us = Us.copy()

            # get cores
            cores = self._get_cores(Xs, Us)
            # print(cores)
            # estimate the coefficients of AR and MA model
            alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)
            for n in range(len(self._Rs)):
                cores_shape = cores[0].shape
                unfold_cores = self._update_cores(n, Us, Xs, es, cores, alpha, beta, lam=1)
                cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
                # update Us
                Us = self._update_Us(Us, Xs, unfold_cores, n)

                for i in range(self._q):
                    # update Es
                    es = self._update_Es(es, alpha, beta, unfold_cores, i, n)

            # convergence check:
            convergence = self._compute_convergence(Us, old_Us)
            con_loss.append(convergence)

            if k % 10 == 0:
                if self._verbose == 1:
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    # print("alpha: {}, beta: {}".format(alpha, beta))

            if self._tol > convergence:
                if self._verbose == 1:
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    print("alpha: {}, beta: {}".format(alpha, beta))
                break

        # Step 3: Forecasting
        # get cores
        cores = self._get_cores(Xs, Us)
        alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)

        new_core = np.sum([al * core for al, core in zip(alpha, cores[-self._p:][::-1])], axis=0) - \
                   np.sum([be * e for be, e in zip(beta, es[-1][::-1])], axis=0)

        new_X = tl.tenalg.multi_mode_dot(new_core, Us)
        Xs.append(new_X)

        if self._d != 0:
            Xs = self._tensor_reverse_diff(self._d, begin, Xs, 0)
        mdt_result = Xs[-1]

        # Step 4: Inverse MDT
        # get orignial shape
        fore_shape = list(self._ts_ori_shape)
        merged = []
        for i in range(trans_data.shape[-1]):
            merged.append(trans_data[..., i].T)
        merged.append(mdt_result.T)

        merged = np.array(merged)

        mdt_result = merged.T

        # 1-step extension (time dimension)
        fore_shape[-1] += 1
        fore_shape = np.array(fore_shape)

        # inverse MDT
        result = self._inverse_MDT(mdt, mdt_result, self._taus, fore_shape)

        return result, con_loss
