# coding: utf-8
# 5-cross validation

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


# =========================================
# Preliminaries


# -----------------------------------------
# Packages

from hparam import default_args_params
from hparam import default_logs_folder
from hparam import default_data_feedin

from hparam import situation_cross_validation
from execute import BK_LOG_LEV
from execute import auxrun_expts, output_starts

from execute import utilise_SAEP, utilise_AdaNet
from execute import run_experiment


# from execute import ensemble_pruning_set
# from execute import experiment_name_set

# from dataset import FEATURES_KEY
# from hparam import super_input_fn
# from dataset import establish_baselines, make_config

# from estimat import AdaNetOriginal, AdaNetVariants
# from estimat import AdaPruOriginal, AdaPruVariants
from classes import PyFile


# -----------------------------------------
# Argparser

logging.basicConfig(level=BK_LOG_LEV)

parser = default_args_params()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
TF_LOG_TLE, LOG_TLE, LOG_DIR = default_logs_folder(args)


# -----------------------------------------
# Parameters

RANDOM_SEED = None
if args.random_seed != 'None':
  RANDOM_SEED = int(args.random_seed)

LEARN_MIXTURE_WEIGHTS = args.adanet_learn_mixture


# -----------------------------------------
# Pruning

thinp_alpha = args.thinp_alpha
type_pruning = args.type_pruning


# -----------------------------------------
# Packages

np.random.seed(RANDOM_SEED)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# for i in range(len(gpus)):
#   tf.config.experimental.set_virtual_device_configuration(
#       gpus[i], [
#           tf.config.experimental.VirtualDeviceConfiguration(
#               memory_limit=2048)])


# =========================================
# Experimental setup

# -----------------------------------------
# Dataset

NUM_CLASS, NUM_SHAPE, _, \
    X_train, y_train, X_test, y_test = default_data_feedin(args)

fed_data = args.dataset
if fed_data.startswith('cifar10'):
  NUM_CLASS = 10
elif fed_data.endswith('mnist'):  # or 'fmnist'
  NUM_CLASS = 10
else:
  raise ValueError("invalid `dataset`, {}.".format(fed_data))


# -----------------------------------------
# Models & Pruning

modeluse = args.model_setting
experiment_name, this_experiment = auxrun_expts(
    type_pruning, thinp_alpha, LEARN_MIXTURE_WEIGHTS, modeluse)
directory = os.path.join(LOG_DIR, this_experiment)


# -----------------------------------------
# Models & Pruning

if type_pruning.startswith('AdaNet'):
  creator = utilise_AdaNet(
      type_pruning, LEARN_MIXTURE_WEIGHTS, modeluse, RANDOM_SEED)
elif type_pruning[:-2] in ['SAEP', 'PRS', 'PAP', 'PIE']:
  creator = utilise_SAEP(
      type_pruning, thinp_alpha,
      LEARN_MIXTURE_WEIGHTS, modeluse, RANDOM_SEED)
else:
  raise ValueError('invalid `type_pruning`, {}.'.format(type_pruning))


# -----------------------------------------
# Parameters

LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
TRAIN_STEPS = args.train_steps

ADANET_LAMBDA = args.adanet_lambda
ADANET_ITERATIONS = args.adanet_iterations

creator.assign_expt_params(NUM_CLASS, this_experiment, LOG_DIR)
creator.assign_train_param(LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS)
creator.assign_adanet_para(
    # ADANET_ITERATIONS, ADANET_LAMBDA, LEARN_MIXTURE_WEIGHTS)
    ADANET_ITERATIONS, ADANET_LAMBDA)
# creator.assign_SAEP_adapru(ensemble_pruning, thinp_alpha, logger)


# =========================================
# Recording

# -----------------------------------------
# Logs

logger = logging.getLogger('adapru')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s: %(levelname)s | %(message)s')

tflog = logging.getLogger('tensorflow')
BK_LOG_TLE = TF_LOG_TLE + '_tf.txt'

if os.path.exists(BK_LOG_TLE):
  os.remove(BK_LOG_TLE)
tf_fh = logging.FileHandler(BK_LOG_TLE)
tf_fh.setLevel(BK_LOG_LEV)

tf_fm = logging.Formatter(logging.BASIC_FORMAT, None)
tf_fh.setFormatter(tf_fm)
tflog.addHandler(tf_fh)


TF_ARCH = 'architecture-{}.json'.format(ADANET_ITERATIONS - 1)
TF_SRCP = os.path.join(LOG_DIR, this_experiment)
TF_FILE = PyFile()

csv_file = open(LOG_TLE + '.csv', 'w', newline="")
csv_writer = csv.writer(csv_file)

if not type_pruning.startswith('AdaNet'):
  creator.assign_SAEP_logger(logger)


# -----------------------------------------
# Auxiliary


# -----------------------------------------
# Single execution

nb_cv = args.cross_validation
if nb_cv <= 1:
  wr_cv = '_sg'  # sing.

  run_experiment(
      X_train, y_train, X_test, y_test, NUM_CLASS,
      NUM_SHAPE, RANDOM_SEED, LOG_TLE, wr_cv,
      logger, formatter, csv_writer,
      creator, modeluse, TF_ARCH, TF_SRCP, TF_LOG_TLE,
      type_pruning, experiment_name, this_experiment,
      LOG_DIR, directory, args)

  sys.exit()


# -----------------------------------------
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
  # creator.assign_SAEP_adapru(ensemble_pruning, thinp_alpha, logger)

  run_experiment(
      X_trn, y_trn, X_tst, y_tst, NUM_CLASS,
      NUM_SHAPE, RANDOM_SEED, LOG_TLE, wr_cv,
      logger, formatter, csv_writer,
      creator, modeluse, TF_ARCH, TF_SRCP, TF_LOG_TLE,
      type_pruning, experiment_name, this_experiment,
      LOG_DIR, directory, args)

  del X_trn, y_trn, X_tst, y_tst


# -----------------------------------------


logger.info("")
discard = os.path.join(os.getcwd(), "*.json")
for fname in glob.glob(discard):
  os.remove(fname)
  logger.info("Deleted " + str(fname))

# discard = glob.glob("*.txt")
# discard.remove("requirements.txt")

csv_file.close()
del csv_writer


# if __name__ == "__main__":
#   pass


# -----------------------------------------

# logger.critical("Saved location:")
# logger.critical("\tLOG_DIR: {:s}".format(LOG_DIR))
# logger.critical("\tLOG_TLE: {:s}".format(LOG_TLE + wr_cv))
# logger.critical("")
# logger.warning(
#   "`cuda_device to use GPU =  {}".format(args.cuda_device))
# logger.warning(
#   "if successful using gpu =  {}".format(tf.test.is_gpu_available()))
# logger.critical("-----------\n")


# -----------------------------------------
# e.g.,


# =========================================
# Parameters


# -----------------------------------------
# Hyper-Parameters
