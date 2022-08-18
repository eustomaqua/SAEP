# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import logging

import os
import shutil
import time

import numpy as np

from estimat import AdaNetOriginal, AdaNetVariants
from estimat import AdaPruOriginal, AdaPruVariants

from dataset import super_input_fn, establish_baselines, FEATURES_KEY

BK_LOG_LEV = logging.INFO
# BK_LOG_LEV = logging.DEBUG
logging.basicConfig(level=BK_LOG_LEV)


# ======================================
# Pruning


ensemble_pruning_set = {
    "AdaNet": "keep_all",
    "SAEP": "keep_all",
    "PRS": "pick_randsear",
    "PAP": "pick_worthful",
    "PIE": "pick_infothin",
}


experiment_name_set = {
    'linear': 'linear',
    'dnn': 'simple_dnn',
    'cnn': 'simple_cnn',
    'cpx': 'complex_cnn',
}


# --------------------------------------
# AdaNet


def utilise_AdaNet(type_pruning='AdaNet.O',
                   learn_mixture_weights=False,
                   modeluse='dnn',
                   random_seed=None):
  assert type_pruning in ['AdaNet.O', 'AdaNet.W']

  assert modeluse in ['linear', 'dnn', 'cnn']
  if type_pruning == 'AdaNet.W':
    assert modeluse != 'linear'

  if type_pruning.endswith('O'):
    creator = AdaNetOriginal(random_seed)
  elif not learn_mixture_weights:
    creator = AdaNetVariants(random_seed)
    creator.LEARN_MIXTURE_WEIGHTS = False
  else:
    creator = AdaNetVariants(random_seed)
    creator.LEARN_MIXTURE_WEIGHTS = True

  return creator


# --------------------------------------
# SAEP


def utilise_SAEP(type_pruning='AdaNet.O',
                 thinp_alpha = 0.5,
                 learn_mixture_weights=False,
                 modeluse='dnn',
                 random_seed=None):
  if type_pruning.startswith('SAEP'):
    type_pruning = type_pruning.replace('SAEP', 'AdaNet')

  assert type_pruning[-1] in ['O', 'W']
  assert type_pruning[:-2] in ['AdaNet', 'PRS', 'PAP', 'PIE']
  assert modeluse in ['dnn', 'cnn', 'cpx']
  # assert modeluse in ['dnn', 'cnn']

  if type_pruning.endswith('O'):
    creator = AdaPruOriginal(random_seed, type_pruning)
  elif not learn_mixture_weights:
    creator = AdaPruVariants(random_seed, type_pruning)
    creator.LEARN_MIXTURE_WEIGHTS = False
  else:
    creator = AdaPruVariants(random_seed, type_pruning)
    creator.LEARN_MIXTURE_WEIGHTS = True

  ens_pruning = type_pruning[:-2]
  if ens_pruning != 'PIE':
    creator.assign_SAEP_adapru(ensemble_pruning_set[ens_pruning])
  else:
    creator.assign_SAEP_adapru(ensemble_pruning_set[ens_pruning],
                               thinp_alpha=thinp_alpha)

  return creator


# ======================================
# Experiment


# --------------------------------------
# Auxilliary


def auxrun_expts(type_pruning, thinp_alpha=.5,
                 lmw=False, modeluse=''):
  experiment_name = experiment_name_set[modeluse]

  this_experiment = os.path.join(experiment_name, type_pruning)
  if type_pruning.endswith('W'):
    this_experiment += lmw
  if type_pruning.startswith('PIE'):
    this_experiment += str(thinp_alpha)

  return experiment_name, this_experiment


# --------------------------------------


def run_experiment(X_trn, y_trn, X_tst, y_tst,
                   NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                   LOG_TLE, wr_cv,
                   logger, formatter,
                   creator, modeluse,
                   output_starts, output_ending, output_arches):
  input_fn = super_input_fn(
      X_trn, y_trn, X_tst, y_tst, NUM_SHAPE, RANDOM_SEED)
  head, feature_columns = establish_baselines(
      NUM_CLASS, NUM_SHAPE, FEATURES_KEY)

  # ' ''
  BK_LOG_TLE = LOG_TLE + wr_cv + '.log'
  if os.path.exists(BK_LOG_TLE):
    os.remove(BK_LOG_TLE)
  log_file = logging.FileHandler(BK_LOG_TLE)
  # log_file.setLevel(logging.DEBUG)
  log_file.setLevel(BK_LOG_LEV)
  log_file.setFormatter(formatter)
  logger.addHandler(log_file)
  # ' ''

  output_starts(logger)
  since = time.time()

  estimator = creator.create_estimator(
      modeluse, feature_columns, head, input_fn)
  results, estimator = creator.train_and_evaluate(
      estimator, input_fn)

  adanet_loss = estimator._adanet_loss
  diver_weight = estimator._diver_weight
  diver_subnet = estimator._diver_subnet

  logger.info("Accuracy: {}".format(results["accuracy"]))
  logger.info("Loss: {}".format(results["average_loss"]))

  ensem_arch = creator.ensemble_architecture(results)
  logger.info("ensemble_architecture: {}".format(ensem_arch))
  logger.info(" adanet_loss: {}".format(np.mean(adanet_loss)))
  logger.info("diver_weight: {}".format(np.mean(diver_weight)))
  logger.info("diver_subnet: {}".format(np.mean(diver_subnet)))

  csv_temp = output_ending(logger, since, wr_cv)

  avg_loss = results["average_loss"]
  accurracy = results["accuracy"]
  diversity = ''

  csv_temp = [
      avg_loss,
      # accurracy,
      # csv_temp,
      # ensem_arch,

      accurracy * 100.,
      np.mean(diver_weight),
      np.mean(diver_subnet),

      # np.std(adanet_loss),
      # np.std(diver_weight),
      # np.std(diver_subnet),

      csv_temp,
  ]

  csv_rows = [[
      'adanet_loss', 'diver_weight', 'diver_subnet',
      '', 'len=', len(adanet_loss)]]
  for i, j, k in zip(adanet_loss, diver_weight, diver_subnet):
    # csv_rows.append(['', '', '', i, j, k])
    csv_rows.append([i, j, k])
  output_arches(logger, wr_cv, csv_temp, csv_rows)
