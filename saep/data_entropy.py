# coding: utf8
# ``data_entorpy.py``
# Aim to: some calculations of entropy (existing methods)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from copy import deepcopy
import time

from numba import njit
from numba import jit, float_

from pathos import multiprocessing as pp
import numpy as np

import tensorflow as tf 
# DELETE = True


# =============================================
#  For SAEP
# =============================================
#


CONST_ZERO = 1e-16


def check_zero(temp):
  return temp if temp != 0. else CONST_ZERO


class PrunedAdanetInfo(object):
  """docstring for PrunedAdanetInfo"""

  def __init__(self, zero=None, num_classes=-1):
    self.CONS_ZERO = 1e-18 if not zero else zero
    self.num_classes = 10 if num_classes == -1 else num_classes

  # @jit(float_(float_))
  @njit
  def info_amount(self, p):
    if p == 0.0:
      return 0.0
    return -1. * np.log(p)

  @njit
  def info_entropy(self, px):
    ans = 0.0
    for i in px:
      tem = self.check_zero(i)
      ans += -1. * i * np.log(tem)
    return ans

  @njit
  def KL_divergence(self, px, py):
    ans = 0.0 
    for i in range(len(px)):
      tem = self.check_zero(px[i] / py[i])  
      ans += px[i] * np.log(tem)
    return ans

  @njit
  def cross_entropy(self, px, py):
    ans = 0.0
    for i in range(len(px)):
      tem = self.check_zero(py[i])
      ans += -1. * px[i] * np.log(tem)
    return ans

  def check_zero(self, tem):
    return tem if tem != 0.0 else self.CONS_ZERO

  # ----  Probability of Discrete Variable  ----

  def prob(self, X):
    # probability of one vector
    X = np.array(X)
    vX = np.arange(self.num_classes).tolist()
    px = np.zeros(self.num_classes)
    for i in range(self.num_classes):
      px[i] = np.mean(X == vX[i])
    px = px.tolist()
    return deepcopy(px)

  def jointProb(self, X, Y):
    # joint probability of two vectors
    X = np.array(X)
    Y = np.array(Y)
    vXY = np.arange(self.num_classes).tolist()
    pxy = np.zeros((self.num_classes, self.num_classes))
    for i in range(self.num_classes):
      for j in range(self.num_classes):
        pxy[i, j] = np.mean((X == vXY[i]) & (Y == vXY[j]))
    pxy = pxy.tolist()
    return deepcopy(pxy)

  # ----------  Shannon Entropy  -----------

  def H(self, p):
    # for a scalar value
    if p == 0.0:
      return 0.0
    return -1. * p * np.log(p) 

  def H1(self, X):
    # H(X), H(Y): for one vector
    px = self.prob(X)
    ans = 0.0
    for i in px:
      ans += self.H(i)
    return ans

  def H2(self, X, Y):
    # H(X,Y): for two vectors
    pxy = self.jointProb(X, Y)
    ans = 0.0 
    for i in pxy:
      for j in i:
        ans += self.H(j)
    return ans

  # ----------  Information Entropy  -----------
  # .. of two discrete random variables

  def I(self, X, Y):
    # I(X;Y): the mutual information function
    px = self.prob(X)
    py = self.prob(Y)
    pxy = self.jointProb(X, Y)
    ans = 0.0
    for i in range(self.num_classes):
      for j in range(self.num_classes):
        tem = self.check_zero(px[i] * py[j])
        tem = self.check_zero(pxy[i][j] / tem)
        ans += pxy[i][j] * np.log(tem)
    return ans

  def MI(self, X, Y):
    # MI(X,Y): the normalized mutual information of ..
    tem = self.H1(X) * self.H1(Y)
    tem = self.check_zero(np.sqrt(tem))
    return self.I(X, Y) / tem

  def VI(self, X, Y):
    # VI(X,Y): the normalized variation of information of ..
    tem = self.check_zero(self.H2(X, Y))
    return 1. - self.I(X, Y) / tem

  def TDAC(self, X, Y, L, lam):
    if X == Y:
      return 0.0 
    tem = self.MI(X, L) + self.MI(Y, L)
    return lam * self.VI(X, Y) + (1. - lam) * tem / 2. 

  def TDAS1(self, S, L, lam):
    S = np.array(S)
    k = S.shape[1]
    ans = [[self.TDAC(S[:, i].tolist(), S[:, j].tolist(), L, lam)
            for j in range(k)] for i in range(k)]
    ans = np.sum(ans) / 2. 
    return ans

  def TDAS2(self, S, L, lam):
    S = np.array(S)
    k = S.shape[1]
    ans1 = [[self.VI(S[:, i].tolist(), S[:, j].tolist())
             for j in range(k)] for i in range(k)]
    ans1 = np.sum(ans1)
    ans2 = [self.MI(S[:, i].tolist(), L) for i in range(k)]
    ans2 = np.sum(ans2)
    return ans1 * lam / 2. + ans2 * (1. - lam) * (k - 1.) / 2.


class PrunedAdanetSelect(PrunedAdanetInfo):
  """docstring for PrunedAdanetSelect"""

  def __init__(self, zero=None, num_classes=-1, random_seed=None):
    self.random_seed = random_seed
    self.prng = np.random
    super(PrunedAdanetSelect, self).__init__(zero=zero, num_classes=num_classes)
    self.TYP_BOL = np.bool
    self.TYP_FLT = np.float32
    self.TYP_INT = np.int32

  # ----------  Convert data  -----------
  # minimum description length

  @njit
  def binsMDL(self, data, nb_bin=5):
    # Let `U' be a set of size `d' of labelled instances accompanied 
    # by a large set of features `N' with cardinality `n', represented
    # in a `dxn' matrix. 
    data = np.array(data, dtype=self.TYP_FLT)
    d = data.shape[0]  # number of samples
    n = data.shape[1]  # number of features
    for j in range(n):
      trans = data[:, j]
      fmin = np.min(trans)
      fmax = np.max(trans)
      fgap = (fmax - fmin) / nb_bin
      idx = (data[:, j] == fmin)
      trans[idx] = 0
      pleft = fmin
      pright = fmin + gap
      for i in range(nb_bin):
        idx = ((data[:, j] > pleft) & (data[:, j] <= pright))
        trans[idx] = i
        pleft += fgap
        pright += fgap
      data[:, j] = deepcopy(trans)
    data = np.array(data, dtype=self.TYP_INT).tolist()
    return deepcopy(data)

  @njit
  def select_features(self, X_trn, y_trn, k1, m1, lam1, X_val, X_tst):
    since = time.time()
    Xd_trn = self.binsMDL(X_trn)  # , nb_bin=5
    if m1 == 1:
      S1 = self.Greedy(Xd_trn, k1, y_trn, lam1)
    else:
      S1 = self.DDisMI(Xd_trn, k1, m1, y_trn, lam1)
    Xs_trn = np.array(X_trn)[:, S1].tolist()
    Xs_tst = np.array(X_tst)[:, S1].tolist()
    Xs_val = np.array(X_val)[:, S1].tolist() if not X_val else []
    S1 = np.where(np.array(Sl) == True)[0].tolist()
    time_elapsed = time.time() - since
    return deepcopy(S1), time_elapsed, deepcopy(Xs_trn), deepcopy(Xs_val), deepcopy(Xs_tst)

  @njit
  def select_weak_cls(self, y_insp, y_trn, k2, m2, lam2, y_cast, y_pred):
    since = time.time()
    yd_insp = np.array(y_insp).T.tolist()
    if m2 == 1:
      S2 = self.Greedy(yd_insp, k2, y_trn, lam2)
    else:
      S2 = self.DDisMI(yd_insp, k2, m2, y_trn, lam2)
    ys_insp = np.array(y_insp)[S2].tolist()
    ys_pred = np.array(y_pred)[S2].tolist()
    ys_cast = np.array(y_cast)[S2].tolist() if not y_cast else []
    S2 = np.where(np.array(S2) == True)[0].tolist()
    time_elapsed = time.time() - since
    return deepcopy(S2), time_elapsed, ys_insp, ys_cast, ys_pred

  # ----------- Algorithm COMEP -----------

  # @njit
  def centralised_OMEP(self, T, k, L, lam):
    r""" Params:
    T:   2D list, set of points/features
    k:   int, number of selected features/individual classifiers
    L:   1D list, target
    lam: float, \lambda
    """

    T = np.array(T)
    n = T.shape[1]
    T = T.tolist()
    S = np.zeros(n, dtype=self.TYP_BOL).tolist()

    p = self.prng.randint(0, n)
    S[p] = True
    for i in range(1, k):
      idx = self.arg_max_p(T, S, L, lam)
      if idx > -1:
        S[idx] = True

    return deepcopy(S)

  # @njit
  def arg_max_p(self, T, S, L, lam):
    r""" Params:
    S:  1D list with elements {True, False}, 
        represents this one is in S or not, and
        S is selected features. 
    """

    T = np.array(T)
    S = np.array(S)
    all_q_in_S = T[:, S].tolist()
    idx_p_not_S = np.where(S == False)[0]
    if len(idx_p_not_S) == 0:
      return -1

    ans = [self.tdac_sum(
        T[:, i].tolist(), all_q_in_S, L, lam) for i in idx_p_not_S]
    idx_p = ans.index(np.max(ans))
    idx = idx_p_not_S[idx_p]
    return idx

  def tdac_sum(self, p, S, L, lam):
    S = np.array(S)
    n = S.shape[1]
    ans = 0.0
    for i in range(n):
      ans += self.TDAC(p, S[:, i].tolist(), L, lam)
    return ans

  # ----------- Algorithm DOMEP -----------

  # @njit
  def distributed_OMEP(self, N, k, m, L, lam):
    r""" Params:
    N:   2D list, set of points/features
    k:   int, number of selected features
    m:   int, number of machines
    L:   1D list, target
    lam: float, \lambda
    """

    N = np.array(N)
    n = N.shape[1]
    Tl = self.randomly_partition(n=n, m=m)
    Tl = np.array(Tl)
    Sl = np.zeros(n, dtype=self.TYP_INT) - 1  # init

    # concurrent selection
    pool = pp.ProcessingPool(nodes = m)
    sub_idx = pool.map(
        self.find_idx_in_sub,
        range(m), [Tl] * m, [N] * m, [k] * m, [L] * m, [lam] * m)

    for i in range(m):
      Sl[sub_idx[i]] = i

    sub_all_in_N = np.where(Sl != -1)[0]
    sub_all_greedy = self.centralised_OMEP(
        N[:, (Sl != -1)].tolist(), k, L, lam)
    sub_all_greedy = np.where(np.array(sub_all_greedy) == True)[0]

    final_S = np.zeros(n, dtype=self.TYP_BOL)
    final_S[sub_all_in_N[sub_all_greedy]] = 1

    div_temS = self.TDAS1(N[:, final_S].tolist(), L, lam)
    div_Sl = [self.TDAS1(
        N[:, (Sl == i)].tolist(), L, lam) for i in range(m)]
    if np.sum(np.array(div_Sl) > div_temS) >= 1:
      tem_argmax_l = div_Sl.index(np.max(div_Sl))
      final_S = (Sl == tem_argmax_l)

    final_S = final_S.tolist()
    return deepcopy(final_S)

  def randomly_partition(self, n, m):
    tem = np.arange(n)
    self.prng.shuffle(tem)
    idx = np.zeros(n, dtype=self.TYP_INT) - 1
    if n % m != 0:
      floors = int(np.floor(n / float(m)))
      ceilings = int(np.ceil(n / float(m)))
      modulus = n - m * floors
      mumble = m * ceilings - n
      # modulus = n % m
      for k in range(modulus):
        ij = tem[k * ceilings: (k + 1) * ceilings]
        idx[ij] = k
      ijt = ceilings * modulus
      for k in range(mumble):
        ij = tem[k * floors + ijt: (k + 1) * floors + ijt]
        idx[ij] = k + modulus
    else:
      ijt = int(n / float(m))
      for k in range(m):
        ij = tem[k * ijt: (k + 1) * ijt]
        idx[ij] = k
    idx = idx.tolist()
    return deepcopy(idx)

  def find_idx_in_sub(self, i, Tl, N, k, L, lam):
    # Group/Machine i-th
    sub_idx_in_N = np.where(Tl == i)[0]
    sub_idx_greedy = self.centralised_OMEP(
        N[:, (Tl == i)].tolist(), k, L, lam)
    sub_idx_greedy = np.where(np.array(sub_idx_greedy) == True)[0]
    ans = sub_idx_in_N[sub_idx_greedy]
    return deepcopy(ans)  # np.ndarray


def contingency_table_multiclass(ha, hb, y):
  a = np.sum(np.logical_and(np.equal(ha, y), np.equal(hb, y)))
  b = np.sum(np.logical_and(np.not_equal(ha, y), np.equal(hb, y)))
  c = np.sum(np.logical_and(np.equal(ha, y), np.not_equal(hb, y)))
  d = np.sum(np.logical_and(np.not_equal(ha, y), np.not_equal(hb, y)))
  # a,b,c,d are `np.integer` (not `int`), a/b/c/d.tolist() gets `int`
  return int(a), int(b), int(c), int(d)


def disagreement_measure_multiclass(ha, hb, y, m):
  _, b, c, _ = contingency_table_multiclass(ha, hb, y)
  return (b + c) / float(m)


def disagreement_measure_without_label(ha, hb, m):
  bc = np.sum(np.not_equal(ha, hb))
  return bc / float(m)


def pairwise_measure_all_disagreement(yt, y, m, nb_cls):
  ans = 0.
  if nb_cls <= 1:
    return ans
  for i in range(nb_cls - 1):
    for j in range(i + 1, nb_cls):
      tem = disagreement_measure_multiclass(yt[i], yt[j], y, m)
      ans += tem
  return ans * 2. / check_zero(nb_cls * (nb_cls - 1.))


def pairwise_measure_all_without_label(yt, m, nb_cls):
  ans = 0.
  if nb_cls <= 1:
    return ans
  for i in range(nb_cls - 1):
    for j in range(i + 1, nb_cls):
      tem = disagreement_measure_without_label(yt[i], yt[j], m)
      ans += tem
  return ans * 2. / check_zero(nb_cls * (nb_cls - 1.))
