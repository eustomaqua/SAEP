# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os
# import shutil
import numpy as np

from saep.test_utils_const import FIXED_SEED
from dataset import FEATURES_KEY, establish_baselines
from hparam import super_input_fn
from execute import utilise_AdaNet, utilise_SAEP
from execute import ensemble_pruning_set, auxrun_expts


# --------------------------------------


prng = np.random.RandomState(FIXED_SEED)
RANDOM_SEED = 10000 - FIXED_SEED

LEARNING_RATE = 0.001
TRAIN_STEPS = 2  # 6000
BATCH_SIZE = 2

ADANET_ITERATIONS = 2  # 11
ADANET_LAMBDA = 0
LEARN_MIXTURE_WEIGHTS = False

LOG_TLE = 'discard'
LOG_DIR = os.path.join(os.getcwd(), "tmpmodels", "discard")
TF_LOG_TLE = 'discard_tf'
logger = None


nb_feat, nb_labl = 7, 4
# nb_trn, nb_tst = 20, 4
nb_trn, nb_tst = 10, 2
nb_shap = (28, 28, 1)

y_trn = prng.randint(nb_labl, size=nb_trn)
y_tst = prng.randint(nb_labl, size=nb_tst)
X_trn = prng.rand(nb_trn, *nb_shap)
X_tst = prng.rand(nb_tst, *nb_shap)

this_experiment = 'casual'


def _run(tp, lmw, um):
  experiment_name, this_experiment = auxrun_expts(
      tp, lmw=lmw, modeluse=um)
  directory = os.path.join(LOG_DIR, this_experiment)
  # if os.path.exists(directory):
  #   shutil.rmtree(directory)

  input_fn = super_input_fn(
      X_trn, y_trn, X_tst, y_tst, nb_shap, RANDOM_SEED)
  head, feature_columns = establish_baselines(
      nb_labl, nb_shap, FEATURES_KEY)
  return experiment_name, this_experiment, \
      input_fn, head, feature_columns


# --------------------------------------


class Test_SAEP(unittest.TestCase):
  def curr(self, tp, lmw=False, um='dnn'):
    if tp.endswith('O'):
      cs = utilise_SAEP(tp, modeluse=um)
    else:
      cs = utilise_SAEP(
          tp, learn_mixture_weights=lmw, modeluse=um)

    exnm, excr, input_fn, head, fcs = _run(tp, lmw, um)
    cs.assign_expt_params(nb_labl, excr, LOG_DIR)
    cs.assign_train_param(LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS)
    cs.assign_adanet_para(ADANET_ITERATIONS, ADANET_LAMBDA)

    ep = tp[:-2]
    cs.assign_SAEP_adapru(ensemble_pruning_set[ep])
    cs.assign_SAEP_logger(logger)
    et = cs.create_estimator(um, fcs, head, input_fn)
    # r, et = cs.train_and_evaluate(et, input_fn)

  def impl(self, tp, lmw=False):
    if tp.endswith('O'):
      self.curr(tp, um='dnn')
      self.curr(tp, um='cnn')
      # self.curr(tp, um='cpx')
      return
    self.curr(tp, lmw, 'dnn')
    self.curr(tp, lmw, 'cnn')
    # self.curr(tp, lmw, 'cpx')

  def test_main(self):
    self.impl('SAEP.O')
    # self.impl('SAEP.W', lmw=False)
    # self.impl('SAEP.W', lmw=True)

    self.impl('AdaNet.O')
    self.impl('AdaNet.W', lmw=False)
    self.impl('AdaNet.W', lmw=True)


class Test_PRS(Test_SAEP):
  def test_main(self):
    self.impl('PRS.O')
    self.impl('PRS.W', lmw=False)
    self.impl('PRS.W', lmw=True)


class Test_PAP(Test_SAEP):
  def test_main(self):
    self.impl('PAP.O')
    self.impl('PAP.W', lmw=False)
    self.impl('PAP.W', lmw=True)


class Test_PIE(Test_SAEP):
  def curr(self, tp, lmw=False, um='dnn'):
    if tp.endswith('O'):
      cs = utilise_SAEP(tp, 0.4, modeluse=um)
    else:
      cs = utilise_SAEP(tp, 0.4, lmw, um)

    _, exnm, input_fn, head, fcs = _run(tp, lmw, um)

    cs.assign_expt_params(nb_labl, exnm, LOG_DIR)
    cs.assign_train_param(LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS)
    cs.assign_adanet_para(ADANET_ITERATIONS, ADANET_LAMBDA)

    ep = tp[:-2]
    cs.assign_SAEP_adapru(ensemble_pruning_set[ep], thinp_alpha=.4)
    cs.assign_SAEP_logger(logger)
    et = cs.create_estimator(um, fcs, head, input_fn)

  def test_main(self):
    self.impl('PIE.O')
    self.impl('PIE.W', lmw=False)
    self.impl('PIE.W', lmw=True)


# --------------------------------------

class Test_adanet(unittest.TestCase):
  def curr(self, tp, lmw=False, um='linear'):
    if tp.endswith('O'):
      cs = utilise_AdaNet(tp, modeluse=um)
    else:
      cs = utilise_AdaNet(tp, lmw, um)

    ex_nm, ex_cr, input_fn, head, fcs = _run(tp, lmw, um)

    cs.assign_expt_params(nb_labl, ex_cr, LOG_DIR)
    cs.assign_train_param(LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS)
    cs.assign_adanet_para(ADANET_ITERATIONS, ADANET_LAMBDA)

    et = cs.create_estimator(um, fcs, head, input_fn)

    # if um == 'linear':
    #   return
    # input_fn = _run(tp, lmw, um)
    # r, et = cs.train_and_evaluate(et, input_fn)

  def impl(self, tp, lmw=False):
    if tp.endswith('O'):
      self.curr(tp, um='dnn')
      self.curr(tp, um='cnn')
      self.curr(tp, um='linear')
      return
    self.curr(tp, lmw, 'dnn')
    self.curr(tp, lmw, 'cnn')

  def test_main(self):
    self.impl('AdaNet.O')
    self.impl('AdaNet.W', False)
    self.impl('AdaNet.W', True)
