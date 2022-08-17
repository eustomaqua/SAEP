# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil


from estimat import AdaNetOriginal, AdaNetVariants
from estimat import AdaPruOriginal, AdaPruVariants


# ======================================
# Pruning


ensemble_pruning_set = {
    "AdaNet": "keep_all",
    "SAEP": "keep_all",
    "PRS": "pick_randsear",
    "PAP": "pick_worthful",
    "PIE": "pick_infothin",
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


def utilise_SAEP(type_pruning='SAEP.O',
                 thinp_alpha = 0.5,
                 learn_mixture_weights=False,
                 modeluse='dnn',
                 random_seed=None):
  assert type_pruning[-1] in ['O', 'W']
  assert type_pruning[:-2] in ['SAEP', 'PRS', 'PAP', 'PIE']
  assert modeluse in ['dnn', 'cnn', 'cpx']

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
# Parameters


# --------------------------------------
# Hyper-Parameters


# ======================================
# Recording


# --------------------------------------
# Auxilliary


# --------------------------------------
# Logs


# ======================================
# Experiment

# --------------------------------------
