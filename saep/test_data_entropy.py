# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import warnings
import numpy as np


from saep.test_utils_const import FIXED_SEED
from saep.data_entropy import PrunedAdanetSelect

warnings.filterwarnings('ignore')

prng = np.random.RandomState(FIXED_SEED)
case = PrunedAdanetSelect()


class TestEntropy(unittest.TestCase):
  def test_entropy(self):
    X, Y = prng.randint(5, size=(2, 17)).tolist()

    self.assertTrue(case.H(0) >= 0)
    self.assertTrue(case.H(1) >= 0)
    self.assertTrue(case.H(0.5) >= 0)

    self.assertTrue(case.H1(X) >= 0)
    self.assertTrue(case.H1(Y) >= 0)
    self.assertTrue(case.H2(X, Y) >= 0)

    self.assertIsInstance(case.I(X, Y), float)
    self.assertIsInstance(case.MI(X, Y), float)
    self.assertIsInstance(case.VI(X, Y), float)

    self.assertAlmostEqual(case.I(X, Y), case.I(Y, X))
    self.assertAlmostEqual(case.MI(X, Y), case.MI(Y, X))
    self.assertAlmostEqual(case.VI(X, Y), case.VI(Y, X))

  def test_normalised(self):
    X, Y = prng.randint(5, size=(2, 17)).tolist()
    L = prng.randint(5, size=17).tolist()
    lam = 0.5
    self.assertIsInstance(case.TDAC(X, Y, L, lam), float)

    S = prng.randint(5, size=(17, 4)).tolist()
    ans1 = case.TDAS1(S, L, lam)
    ans2 = case.TDAS2(S, L, lam)
    self.assertAlmostEqual(ans1, ans2)

  def test_centralised(self):
    L = prng.randint(5, size=17).tolist()
    lam = 0.5

    T = prng.randint(5, size=(17, 4)).tolist()
    k = 2
    S = case.centralised_OMEP(T, k, L, lam)
    self.assertEqual(np.sum(S), k)

    idx = case.arg_max_p(T, S, L, lam)
    self.assertIsInstance(idx, np.integer)
    T = np.array(T)
    p = T[:, 0].tolist()
    S = T[:, 1:].tolist()
    ans = case.tdac_sum(p, S, L, lam)
    self.assertIsInstance(ans, float)
    del T, k, S, idx, p, ans

  def test_distributed(self):
    L = prng.randint(5, size=17).tolist()
    lam = 0.5

    N = prng.randint(5, size=(17, 6)).tolist()
    k, m = 2, 2
    S = case.distributed_OMEP(N, k, m, L, lam)
    self.assertEqual(np.sum(S), k)

    Tl = case.randomly_partition(6, m)
    Tl = np.array(Tl)
    i = prng.choice(np.unique(Tl))
    N = np.array(N)
    ans = case.find_idx_in_sub(i, Tl, N, k, L, lam)
    self.assertIsInstance(ans, np.ndarray)
    self.assertIsInstance(ans[0], np.integer)
    del N, k, m, S, Tl, i, ans


class TestDiversity(unittest.TestCase):
  # Disagreement measure is used here

  def test_contingency(self):
    from saep.data_entropy import contingency_table_multiclass
    m = 17

    ha = prng.randint(4, size=m).tolist()
    hb = prng.randint(4, size=m).tolist()
    y = prng.randint(4, size=m).tolist()

    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    self.assertTrue(a + b + c + d == m)

  def test_disagreement(self):
    from saep.data_entropy import (
        disagreement_measure_multiclass,
        disagreement_measure_without_label,)

    m = 17

    ha = prng.randint(4, size=m).tolist()
    hb = prng.randint(4, size=m).tolist()
    y = prng.randint(4, size=m).tolist()

    ans = disagreement_measure_multiclass(ha, hb, y, m)
    self.assertTrue(0 <= ans <= 1)

    res = disagreement_measure_without_label(ha, hb, m)
    self.assertTrue(0 <= res <= 1)
    self.assertTrue(ans <= res)

  def test_pairwise(self):
    from saep.data_entropy import (
        pairwise_measure_all_disagreement,
        pairwise_measure_all_without_label)

    m, n = 17, 3

    yt = prng.randint(4, size=(n, m)).tolist()
    y = prng.randint(4, size=m).tolist()

    ans = pairwise_measure_all_disagreement(yt, y, m, n)
    self.assertTrue(0 <= ans <= 1)

    res = pairwise_measure_all_without_label(yt, m, n)
    self.assertTrue(0 <= res <= 1)
    self.assertTrue(ans <= res)
