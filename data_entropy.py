# coding: utf8

# ``data_entorpy.py``
# Aim to: some calculations of entropy (existing methods)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from copy import deepcopy
import gc
import sys
import time

from numba import njit
from numba import jit, float_
import numpy as np 
from scipy import stats

import tensorflow as tf 
gc.enable()

# from pathos import multiprocessing as pp
# from pympler.asizeof import asizeof

DELETE = True


# ----------  Convert data  -----------


# minimum description length
def binsMDL(data, nb_bin=5):  # bins5MDL
  # Let `U' be a set of size `d' of labelled instances
  # accompanied by a large set of features `N' with cardinality `n',
  # represented in a `dxn' matrix. 

  data = np.array(data, dtype=np.float32)
  d = data.shape[0]  # number of samples
  n = data.shape[1]  # number of features

  for j in range(n):  # By Feature
    fmin = np.min(data[:, j])
    fmax = np.max(data[:, j])
    fgap = (fmax - fmin) / nb_bin
    trans = data[:, j]

    idx = (data[:, j] == fmin)
    trans[idx] = 0
    pleft = fmin
    pright = fmin + fgap

    for i in range(nb_bin):
      idx = ((data[:, j] > pleft) & (data[:, j] <= pright))
      trans[idx] = i
      pleft += fgap
      pright += fgap

    data[:, j] = deepcopy(trans)

  data = np.array(data, dtype=np.int8).tolist()
  del d, n, i, j, fmin, fmax, fgap, trans, idx, pleft, pright
  gc.collect()
  return deepcopy(data)  # data  #list


# ----------  Probability of Discrete Variable  -----------


# probability of one vector
def prob(X):
  X = np.array(X)
  vX = np.unique(X).tolist()
  dX = len(vX)
  px = np.zeros(dX)
  for i in range(dX):
    px[i] = np.mean(X == vX[i])
  px = px.tolist()
  del i, X, dX
  gc.collect()
  return deepcopy(px), deepcopy(vX)  # list


# joint probability of two vectors
def jointProb(X, Y):
  X = np.array(X)
  Y = np.array(Y)
  vX = np.unique(X).tolist()
  vY = np.unique(Y).tolist()
  dX = len(vX)
  dY = len(vY)
  pxy = np.zeros((dX, dY))
  for i in range(dX):
    for j in range(dY):
      pxy[i, j] = np.mean((X == vX[i]) & (Y == vY[j]))
  pxy = pxy.tolist()
  del dX, dY, i, j, X, Y
  gc.collect()
  return deepcopy(pxy), deepcopy(vX), deepcopy(vY)  # list


# ----------  Shannon Entropy  -----------
# calculate values of entropy
# H(.) is the entropy function and p(.,.) is the joint probability


# for a scalar value
@jit
def H(p):
  if p == 0.:
    return 0.
  return (-1.) * p * np.log2(p)


# H(X), H(Y) :  for one vector
def H1(X):
  px, _ = prob(X)
  ans = 0.
  for i in px:
    ans += H(i)

  i = -1
  del px, i
  gc.collect()
  return ans


# H(X,Y) :  for two vectors
def H2(X, Y):
  pxy, _, _ = jointProb(X, Y)
  ans = 0.
  for i in pxy:
    for j in i:
      ans += H(j)

  i = j = -1
  del pxy, i, j
  gc.collect()
  return ans


# =====================================
#  Inspired by zadeh2017diversity
# =====================================


# I(.;.) is the mutual information function
# I(X;Y)
def I(X, Y):
  px, _ = prob(X)
  py, _ = prob(Y)
  pxy, _, _ = jointProb(X, Y)

  ans = 0.
  for i in range(len(px)):
    for j in range(len(py)):
      if pxy[i][j] == 0.:
        ans += 0.
      else:
        ans += pxy[i][j] * np.log2(pxy[i][j] / px[i] / py[j])

  i = j = -1
  del px, py, pxy, i, j
  gc.collect()
  return ans


# MI(X,Y):
# The normalized mutual information of two discrete random variables X and Y
def MI(X, Y):
  # return I(X, Y) / np.sqrt(H1(X) * H1(Y))
  tem = np.sqrt(H1(X) * H1(Y))
  ans = I(X, Y) / np.max([tem, 1e-18])
  return ans


# VI(X,Y):
# The normalized variation of information of two discrete random variables X and Y
def VI(X, Y):
  return 1. - I(X, Y) / np.max([H2(X, Y), 1e-18])


# For two feature vectors like p and q, and the class label vector L,
# define DIST(p,q) as follows:
def DIST(X, Y, L, lam):  # lambda
  if X == Y:  # list
    return 0.
  return lam * VI(X, Y) + (1. - lam) * (MI(X, L) + MI(Y, L)) / 2.


# S \subset or \subseteq N,  N is the set of all features and |S|=k.
# We want to maximize the following objective function (as the objective
# of diversity maximization problem) for `S' \subset `N' and |S|=k

def DIV1(S, L, lam):
  S = np.array(S)
  k = S.shape[1]
  ans = [[DIST(S[:, i].tolist(), S[:, j].tolist(), L, lam) for j in range(k)] for i in range(k)]
  ans = np.sum(ans) / 2.
  del S, k
  gc.collect()
  return ans


def DIV2(S, L, lam):
  S = np.array(S)
  k = S.shape[1]
  ans1 = [[VI(S[:, i].tolist(), S[:, j].tolist()) for j in range(k)] for i in range(k)]
  ans1 = np.sum(ans1)
  ans2 = [MI(S[:, i].tolist(), L) for i in range(k)]
  ans2 = np.sum(ans2)
  ans = ans1 * lam / 2. + ans2 * (1. - lam) * (k - 1.) / 2.
  del S, k, ans1, ans2
  gc.collect()
  return ans


# ----------  Algorithm Greedy  -----------


def dist_sum(p, S, L, lam):
  S = np.array(S)
  n = S.shape[1]
  # calc 1
  ans = 0.
  for i in range(n):
    ans += DIST(p, S[:, i].tolist(), L, lam)
    #print("i=",i, "DIST=", DIST(p, S[:,i].tolist(), L, lam))
  del S, n, i
  gc.collect()
  return ans


# T is the set of points/features; S = [True,False] represents this one
# is in S or not, and S is selected features.
def arg_max_p(T, S, L, lam):
  T = np.array(T)
  S = np.array(S)

  # calc 3
  all_q_in_S = T[:, S].tolist()
  idx_p_not_S = np.where(S == False)[0]
  if len(idx_p_not_S) == 0:
    del T, S, all_q_in_S, idx_p_not_S
    return -1  # idx = -1

  ans = [dist_sum(T[:, i].tolist(), all_q_in_S, L, lam) for i in idx_p_not_S]
  idx_p = ans.index(np.max(ans))
  idx = idx_p_not_S[idx_p]

  del T, S, all_q_in_S, idx_p_not_S, idx_p, ans
  gc.collect()
  return idx


# T:    set of points/features
# k:    number of selected features
def Greedy(T, k, L, lam):
  T = np.array(T)
  n = T.shape[1]
  S = np.zeros(n, dtype=np.bool)
  p = np.random.randint(0, n)
  S[p] = True
  for i in range(1, k):
    idx = arg_max_p(T, S, L, lam)
    if idx > -1:
      S[idx] = True  # 1
  S = S.tolist()
  del T, n, p,  # i,idx
  gc.collect()
  return deepcopy(S)  # S  #list


# ----------  Algorithm DDisMI  -----------


def choose_proper_platform(nb, pr):
  m = int(np.round(np.sqrt(1. / pr)))
  k = np.max([int(np.round(nb * pr)), 1])
  while k * m >= nb:
    m = np.max([m - 1, 1])
    if m == 1:
      break
  #m = np.max([m, 2])
  return k, m


def randomly_partition(n, m):
  # randseed = int(time.time() * 1e6 % (2**32-1))
  # prng = np.random.RandomState(randseed)
  tem = np.arange(n)
  # prng.shuffle(tem)
  np.random.shuffle(tem)
  idx = np.zeros(n, dtype=np.int8) - 1  # init  #np.ones(n)-2

  if n % m != 0:
    # 底和顶 floors and ceilings
    floors = int(np.floor(n / float(m)))
    ceilings = int(np.ceil(n / float(m)))
    # 模：二元运算 modulus and mumble 含糊说话
    modulus = n - m * floors
    mumble = m * ceilings - n
    # mod:  n % m

    for k in range(modulus):
      ij = tem[k * ceilings: (k + 1) * ceilings]
      idx[ij] = k
    ijt = ceilings * modulus
    for k in range(mumble):
      ij = tem[k * floors + ijt: (k + 1) * floors + ijt]
      idx[ij] = k + modulus

    del floors, ceilings, modulus, mumble, k, ij, ijt

  else:
    ijt = int(n / m)
    for k in range(m):
      ij = tem[k * ijt: (k + 1) * ijt]
      idx[ij] = k
    del ijt, ij, k

  idx = idx.tolist()
  gc.collect()
  return deepcopy(idx)


# Group/Machine i-th
def find_idx_in_sub(i, Tl, N, k, L, lam):
  sub_idx_in_N = np.where(Tl == i)[0]  # or np.argwhere(Tl == i).T[0]
  sub_idx_greedy = Greedy(N[:, (Tl == i)].tolist(), k, L, lam)
  sub_idx_greedy = np.where(np.array(sub_idx_greedy) == True)[0]
  ans = sub_idx_in_N[sub_idx_greedy]
  del sub_idx_in_N, sub_idx_greedy 
  gc.collect()
  return deepcopy(ans)  # np.array


def DDisMI(N, k, m, L, lam):
  N = np.array(N)
  n = N.shape[1]
  Tl = randomly_partition(n=n, m=m)
  Tl = np.array(Tl)
  Sl = np.zeros(n, dtype=np.int8) - 1  # init

  # define lambda function

  # concurrent selection 
  pool = pp.ProcessingPool(nodes = m)
  sub_idx = pool.map(find_idx_in_sub, range(m), [Tl] * m, [N] * m, [k] * m, [L] * m, [lam] * m)
  del pool, Tl

  for i in range(m):
    Sl[sub_idx[i]] = i 
  del sub_idx
  sub_all_in_N = np.where(Sl != -1)[0]
  sub_all_greedy = Greedy(N[:, (Sl != -1)].tolist(), k, L, lam)
  sub_all_greedy = np.where(np.array(sub_all_greedy) == True)[0]

  final_S = np.zeros(n, dtype=np.bool)
  final_S[sub_all_in_N[sub_all_greedy]] = 1
  del sub_all_in_N, sub_all_greedy

  div_temS = DIV1(N[:, final_S].tolist(), L, lam)
  div_Sl = [DIV1(N[:, (Sl == i)].tolist(), L, lam) for i in range(m)]
  if np.sum(np.array(div_Sl) > div_temS) >= 1:
    tem_argmax_l = div_Sl.index(np.max(div_Sl))
    final_S = (Sl == tem_argmax_l)
    del tem_argmax_l

  del div_temS, div_Sl, N, n, m, Sl
  final_S = final_S.tolist()
  gc.collect()
  return deepcopy(final_S)


# If you want to do ``Serial Execution'', just to do:
# S = Greedy(N, k, L, lam)


# =====================================
#  Inspired by margineantu1997pruning
# =====================================
# KL distance between two probability distributions p and q:
# KL_distance = scipy.stats.entropy(p,q)


# the KL distance between two probability distributions
# p and q is: D(p||q)=

# But the meaning of this function is not quite the same as KL distance
@jit
def KLD(p, q):
  ans = 0.
  n = len(p)  # =len(q)
  for i in range(n):
    #ans += p[i] * np.log2(p[i] / q[i])
    ans += p[i] * np.log2(np.max([p[i], 1e-18]) / np.max([q[i], 1e-18]))
  return ans


# softmax regression
@jit
def softmax(y):
  return np.exp(y) / np.sum(np.exp(y), axis=0)


# =============================================
#  Valuation Codes
# =============================================


class PrunedAdanetInfo(object):
  """docstring for PrunedAdanetInfo"""

  def __init__(self, zero=None, num_classes=-1):
    # if not zero:
    #     self.CONS_ZERO = 1e-18
    # else:
    #     self.CONS_ZERO = zero
    self.CONS_ZERO = 1e-18 if not zero else zero
    self.num_classes = 10 if num_classes == -1 else num_classes

  # @jit(float_(float_))
  @njit
  def info_amount(self, p):
    if p == 0.0:
      return 0.0
    return -1. * np.log(p)
  # @jit(float_(float_))

  @njit
  def info_entropy(self, px):
    ans = 0.0
    for i in px:
      tem = self.check_zero(i)
      ans += -1. * i * np.log(tem)
    return ans
  # @jit(float_(float_,float_))

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

  # @njit
  def prob(self, X):
    X = np.array(X)
    vX = np.arange(self.num_classes).tolist()
    px = np.zeros(self.num_classes)
    for i in range(self.num_classes):
      px[i] = np.mean(X == vX[i])
    px = px.tolist()
    return deepcopy(px)
  # @njit

  def jointProb(self, X, Y):
    X = np.array(X)
    Y = np.array(Y)
    vXY = np.arange(self.num_classes).tolist()
    pxy = np.zeros((self.num_classes, self.num_classes))
    for i in range(self.num_classes):
      for j in range(self.num_classes):
        pxy[i, j] = np.mean((X == vXY[i]) & (Y == vXY[j]))
    pxy = pxy.tolist()
    return deepcopy(pxy)

  # @njit
  def H(self, p):
    if p == 0.0:
      return 0.0
    return -1. * p * np.log(p) 
  # @njit

  def H1(self, X):
    px = self.prob(X)
    ans = 0.0
    for i in px:
      ans += self.H(i)
    return ans
  # @njit

  def H2(self, X, Y):
    pxy = self.jointProb(X, Y)
    ans = 0.0 
    for i in pxy:
      for j in i:
        ans += self.H(j)
    return ans

  # @njit
  def I(self, X, Y):
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
  # @njit

  def MI(self, X, Y):
    tem = self.H1(X) * self.H1(Y)
    tem = self.check_zero(np.sqrt(tem))
    return self.I(X, Y) / tem
  # @njit

  def VI(self, X, Y):
    tem = self.check_zero(self.H2(X, Y))
    return 1. - self.I(X, Y) / tem

  # @njit
  def DIST(self, X, Y, L, lam):
    if X == Y:
      return 0.0 
    tem = self.MI(X, L) + self.MI(Y, L)
    return lam * self.VI(X, Y) + (1. - lam) * tem / 2. 
  # @njit

  def DIV1(self, S, L, lam):
    S = np.array(S)
    k = S.shape[1]
    ans = [[self.DIST(S[:, i].tolist(), S[:, j].tolist(), L, lam)
            for j in range(k)] for i in range(k)]
    ans = np.sum(ans) / 2. 
    return ans
  # @njit

  def DIV2(self, S, L, lam):
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

  @njit
  def binsMDL(self, data, nb_bin=5):
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

  # ----------- Greedy -----------

  # @njit
  def Greedy(self, T, k, L, lam):
    r""" Params:
    T:   2D list, set of points/features
    k:   int, number of selected features
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

  # @jit(forceobj=True)
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
    ans = [self.dist_sum(T[:, i].tolist(), all_q_in_S, L, lam) for i in idx_p_not_S]
    idx_p = ans.index(np.max(ans))
    idx = idx_p_not_S[idx_p]
    return idx

  # @njit
  def dist_sum(self, p, S, L, lam):
    S = np.array(S)
    n = S.shape[1]
    ans = 0.0
    for i in range(n):
      ans += self.DIST(p, S[:, i].tolist(), L, lam)
    return ans

  # ----------- DDisMI -----------

  # @njit
  def DDisMI(self, N, k, m, L, lam):
    """ Params:
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
    sub_idx = pool.map(self.find_idx_in_sub, range(m), [Tl] * m, [N] * m, [k] * m, [L] * m, [lam] * m)
    # 
    for i in range(m):
      Sl[sub_idx[i]] = i
    #
    sub_all_in_N = np.where(Sl != -1)[0]
    sub_all_greedy = self.Greedy(N[:, (Sl != -1)].tolist(), k, L, lam)
    sub_all_greedy = np.where(np.array(sub_all_greedy) == True)[0]
    # 
    final_S = np.zeros(n, dtype=self.TYP_BOL)
    final_S[sub_all_in_N[sub_all_greedy]] = 1
    # 
    div_temS = self.DIV1(N[:, final_S].tolist(), L, lam)
    div_Sl = [self.DIV1(N[:, (Sl == i)].tolist(), L, lam) for i in range(m)]
    if np.sum(np.array(div_Sl) > div_temS) >= 1:
      tem_argmax_l = div_Sl.index(np.max(div_Sl))
      final_S = (Sl == tem_argmax_l)
    #
    final_S = final_S.tolist()
    return deepcopy(final_S)

  # @njit
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

  # @njit
  def find_idx_in_sub(self, i, Tl, N, k, L, lam):
    sub_idx_in_N = np.where(Tl == i)[0]
    sub_idx_greedy = self.Greedy(N[:, (Tl == i)].tolist(), k, L, lam)
    sub_idx_greedy = np.where(np.array(sub_idx_greedy) == True)[0]
    ans = sub_idx_in_N[sub_idx_greedy]
    return deepcopy(ans)  # np.ndarray


CONST_ZERO = 1e-16


def check_zero(temp):
  return temp if temp != 0. else CONST_ZERO


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


def pairwise_measure_all_disagreement(yt, y, m, nb_cls):
  ans = 0.
  if nb_cls <= 1:
    return ans
  for i in range(nb_cls - 1):
    for j in range(i + 1, nb_cls):
      tem = disagreement_measure_multiclass(yt[i], yt[j], y, m)
      ans += tem
  return ans * 2. / check_zero(nb_cls * (nb_cls - 1.))


def disagreement_measure_without_label(ha, hb, m):
  bc = np.sum(np.not_equal(ha, hb))
  return bc / float(m)


def pairwise_measure_all_without_label(yt, m, nb_cls):
  ans = 0.
  if nb_cls <= 1:
    return ans
  for i in range(nb_cls - 1):
    for j in range(i + 1, nb_cls):
      tem = disagreement_measure_without_label(yt[i], yt[j], m)
      ans += tem
  return ans * 2. / check_zero(nb_cls * (nb_cls - 1.))
