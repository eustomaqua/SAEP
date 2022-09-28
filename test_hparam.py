# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from hparam import feed_dataset_all_in
from saep.test_utils_const import FIXED_SEED
prng = np.random.RandomState(FIXED_SEED)


class TestHparams(unittest.TestCase):
  def test_split_data(self):
    from hparam import situation_cross_validation
    nb_iter = 2  # or 3, 5

    nb_inst, nb_labl, nb_feat = 100, 4, 7
    y_unique = list(range(nb_labl))
    y = prng.randint(nb_labl, size=nb_inst)

    split_type = 'cross_valid_v2'
    split_idx = situation_cross_validation(nb_iter, y, split_type)
    for i_trn, i_tst in split_idx:
      self.assertTrue(
          all(np.unique(y[i_trn]) == np.unique(y[i_tst])))
      self.assertTrue(all(np.unique(y[i_trn]) == y_unique))
      self.assertTrue(all(np.unique(y[i_tst]) == y_unique))

    split_type = 'cross_valid_v3'
    split_idx = situation_cross_validation(nb_iter, y, split_type)
    for i_trn, i_val, i_tst in split_idx:
      self.assertTrue(all(np.unique(y[i_trn]) == y_unique))
      self.assertTrue(all(np.unique(y[i_val]) == y_unique))
      self.assertTrue(all(np.unique(y[i_tst]) == y_unique))


class TestClasses(unittest.TestCase):
  def test_py_file(self):
    from classes import PyFile
    import os

    dstpath = os.getcwd()  # './'
    srcpath = os.path.join(dstpath, 'saep')  # './saep'
    filename = 'test_utils_const.py'

    case = PyFile()
    res = case.find_architecture(filename, srcpath)

    res = os.path.join(srcpath, res)
    self.assertTrue(os.path.exists(res))

    dstname = 'discard_'
    case.copy_architecture(filename, srcpath, dstpath, dstname)

    dstname += filename
    res = os.path.join(dstpath, dstname)
    self.assertTrue(os.path.exists(res))
    os.remove(res)


class TestDataset(unittest.TestCase):
  def test_load_cifar(self):
    # from dataset import data_to_feed_in

    fed_data = 'cifar10'
    nc, ns, _, \
        X_trn, y_trn, X_tst, y_tst = feed_dataset_all_in(fed_data)

    self.assertEqual(nc, 10)
    self.assertEqual(ns, (32, 32, 3))

    self.assertEqual(ns, X_trn.shape[1:])
    self.assertEqual(ns, X_tst.shape[1:])

    self.assertTrue(X_trn.shape[0] == y_trn.shape[0] == 50000)
    self.assertTrue(X_tst.shape[0] == y_tst.shape[0] == 10000)

    nc, ns, _, _, y_trn, _, y_tst = feed_dataset_all_in(fed_data, True)

    y_unique = np.unique(np.concatenate([y_trn, y_tst]))
    self.assertTrue(4 in y_unique)  # args.label_zero
    self.assertTrue(9 in y_unique)  # args.label_one

  def test_load_mnist(self):
    # from dataset import data_to_feed_in

    for fed_data in ['mnist', 'fmnist']:
      nc, ns, _, \
          X_trn, y_trn, X_tst, y_tst = feed_dataset_all_in(fed_data)

      self.assertEqual(nc, 10)
      self.assertEqual(ns, (28, 28, 1))

      self.assertEqual(ns, X_trn.shape[1:])
      self.assertEqual(ns, X_tst.shape[1:])

      self.assertTrue(X_trn.shape[0] == y_trn.shape[0] == 60000)
      self.assertTrue(X_tst.shape[0] == y_tst.shape[0] == 10000)
