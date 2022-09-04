# coding: utf-8
# 5-cross validation or single execution

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import csv
import glob
import time

import os
import shutil
import sys

import numpy as np
import tensorflow.compat.v1 as tf


# --------------------------------------
# Packages


from hparam import (default_args, default_logs, default_feed,
                    situation_cross_validation)
from execute import (utilise_SAEP, auxrun_expts, output_starts,
                     run_experiment, BK_LOG_LEV)
# from classes import PyFile


# ======================================
# Hyper-parameters


# --------------------------------------
# Argparser

logging.basicConfig(level=BK_LOG_LEV)

args = default_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
TF_LOG_TLE, LOG_TLE, LOG_DIR = default_logs(args)


# --------------------------------------
# Parameters

RANDOM_SEED = None
if args.random_seed != 'None':
  RANDOM_SEED = int(args.random_seed)

# LEARN_MIXTURE_WEIGHTS = False
# if args.adanet_learn_mixture == 'T':
#   LEARN_MIXTURE_WEIGHTS = True
LEARN_MIXTURE_WEIGHTS = args.adanet_learn_mixture


# --------------------------------------
# Pruning

thinp_alpha = args.thinp_alpha
type_pruning = args.type_pruning
if 'AdaNet' in type_pruning:
  type_pruning = type_pruning.replace('AdaNet', 'SAEP')

# Note that `SAEP` is actually AdaNet just using `saep` to implement,
# in order to get some values to compare with PRS/PAP/PIE.


# --------------------------------------
# Packages

np.random.seed(RANDOM_SEED)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


# ======================================
# Experiment setup


# --------------------------------------
# Dataset

NUM_SHAPE, _, X_train, y_train, X_test, y_test = default_feed(args)

fed_data = args.dataset
if fed_data.startswith('cifar10'):
  NUM_CLASS = 10
elif fed_data.endswith('mnist'):  # or 'fmnist'
  NUM_CLASS = 10
else:
  raise ValueError('invalid `dataset`.')

# if args.binary:
#   NUM_CLASS = 2


# --------------------------------------
# Models & Pruning

modeluse = args.model_setting

experiment_name, this_experiment = auxrun_expts(
    type_pruning, thinp_alpha, LEARN_MIXTURE_WEIGHTS,
    modeluse)
directory = os.path.join(LOG_DIR, this_experiment)


'''
if type_pruning.startswith('AdaNet'):
  creator = utilise_AdaNet(type_pruning,
                           LEARN_MIXTURE_WEIGHTS,
                           modeluse, RANDOM_SEED)
elif type_pruning[:-2] in ['SAEP', 'PRS', 'PAP', 'PIE']:
  creator = utilise_SAEP(type_pruning, thinp_alpha,
                         LEARN_MIXTURE_WEIGHTS,
                         modeluse, RANDOM_SEED)
else:
  raise ValueError("invalid `type_pruning`.")
'''

if type_pruning[:-2] not in ['SAEP', 'PRS', 'PAP', 'PIE']:
  raise ValueError("`type_pruning` invalid, {}."
                   "".format(type_pruning[:-2]))
creator = utilise_SAEP(type_pruning, thinp_alpha,
                       LEARN_MIXTURE_WEIGHTS,
                       modeluse, RANDOM_SEED)


# --------------------------------------
# Parameters

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
TRAIN_STEPS = args.train_steps

ADANET_LAMBDA = args.adanet_lambda
ADANET_ITERATIONS = args.adanet_iterations


creator.assign_expt_params(NUM_CLASS, this_experiment, LOG_DIR)
creator.assign_train_param(LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS)
creator.assign_adanet_para(ADANET_ITERATIONS, ADANET_LAMBDA)


# ======================================
# Recording

# --------------------------------------
# Logs

logger = logging.getLogger('adapru')
formatter = logging.Formatter(
    # '%(asctime)s - %(name)s:%(levelname)s | %(message)s'
    '%(levelname)s | %(message)s')

tflog = logging.getLogger('tensorflow')
BK_LOG_TLE = TF_LOG_TLE + '_tf.txt'
if os.path.exists(BK_LOG_TLE):
  os.remove(BK_LOG_TLE)
tf_fh = logging.FileHandler(BK_LOG_TLE)
# tf_fh.setLevel(logging.DEBUG)
tf_fh.setLevel(BK_LOG_LEV)
tf_fm = logging.Formatter(logging.BASIC_FORMAT, None)
tf_fh.setFormatter(tf_fm)
tflog.addHandler(tf_fh)

TF_ARCH = 'architecture-{}.json'.format(ADANET_ITERATIONS - 1)
TF_SRCP = os.path.join(LOG_DIR, this_experiment)
# TF_FILE = PyFile()

csv_file = open(LOG_TLE + '.csv', 'w', newline="")
csv_writer = csv.writer(csv_file)

creator.assign_SAEP_logger(logger)


# --------------------------------------
# Auxilliary


# --------------------------------------
# Single execution


nb_cv = args.cross_validation

if nb_cv <= 1:
  wr_cv = '_sg'  # sing.

  run_experiment(X_train, y_train, X_test, y_test,
                 NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                 LOG_TLE, wr_cv, logger, formatter, csv_writer,
                 creator, modeluse, TF_ARCH, TF_SRCP,
                 TF_LOG_TLE, type_pruning,
                 experiment_name, this_experiment,
                 LOG_DIR, directory, args)

  sys.exit()


# --------------------------------------
# Cross Validation


X_dataset = np.concatenate([X_train, X_test], axis=0)
y_dataset = np.concatenate([y_train, y_test], axis=0)
del X_train, y_train, X_test, y_test


split_idx = situation_cross_validation(nb_cv, y_dataset)
for i in range(nb_cv):
  idx_trn, idx_tst = split_idx[i]

  X_trn = X_dataset[idx_trn]
  y_trn = y_dataset[idx_trn]
  X_tst = X_dataset[idx_tst]
  y_tst = y_dataset[idx_tst]
  del idx_trn, idx_tst

  wr_cv = "_cv" + str(i + 1)
  run_experiment(X_trn, y_trn, X_tst, y_tst,
                 NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                 LOG_TLE, wr_cv, logger, formatter, csv_writer,
                 creator, modeluse, TF_ARCH, TF_SRCP,
                 TF_LOG_TLE, type_pruning,
                 experiment_name, this_experiment,
                 LOG_DIR, directory, args)
  del X_trn, y_trn, X_tst, y_tst


# -----------------------------------------


logger.info("")
discard = os.path.join(os.getcwd(), "*.json")
for fname in glob.glob(discard):
  os.remove(fname)
  logger.info("Deleted " + str(fname))

csv_file.close()
del csv_writer


# if __name__ == "__main__":
#   pass


# python main.py -cv 1
# python main.py -cv 2 -bi
# python.main.py
