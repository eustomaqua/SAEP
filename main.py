# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import csv

import os
import shutil
import sys

import time

import numpy as np
import tensorflow.compat.v1 as tf


# --------------------------------------
# Argparser

from hparam import default_args, default_logs
from dataset import ensemble_pruning_set

from dataset import FEATURES_KEY, super_input_fn
from dataset import establish_baselines, make_config
from dataset import data_to_feed_in

from estimat import AdaNetOriginal, AdaNetVariants
from estimat import AdaPruOriginal, AdaPruVariants

from classes import PyFile
from hparam import situation_cross_validation


# ======================================
# Preliminaries


# --------------------------------------
# Hyper-Parameters

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('test')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s: %(levelname)s | %(message)s')


# --------------------------------------
# Argparser

parser = default_args()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
TF_LOG_TLE, LOG_TLE, LOG_DIR = default_logs(args)


# --------------------------------------
# Parameters

RANDOM_SEED = None if args.random_seed == 'None' else int(args.random_seed)
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
TRAIN_STEPS = args.train_steps

LEARN_MIXTURE_WEIGHTS = True if args.adanet_learn_mixture == 'T' else False
ADANET_LAMBDA = args.adanet_lambda
ADANET_ITERATIONS = args.adanet_iterations

thinp_alpha = args.thinp_alpha
type_pruning = args.type_pruning


# --------------------------------------
# Packages

np.random.seed(RANDOM_SEED)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


# --------------------------------------
# Data set

NUM_CLASS, NUM_SHAPE, _, X_train, y_train, X_test, y_test = \
    data_to_feed_in(args.dataset,
                    args.binary, args.label_zero, args.label_one)
nb_cv = args.cross_validation


# ======================================
# Parameters


# --------------------------------------
# Logs

tflog = logging.getLogger('tensorflow')
if os.path.exists(TF_LOG_TLE + '.txt'):
  os.remove(TF_LOG_TLE + '.txt')

tf_fh = logging.FileHandler(TF_LOG_TLE + '.txt')
tf_fh.setLevel(logging.DEBUG)

tf_fm = logging.Formatter(logging.BASIC_FORMAT, None)
tf_fh.setFormatter(tf_fm)

tflog.addHandler(tf_fh)


# --------------------------------------
# Model

modeluse = args.model_setting
if modeluse == 'linear':
  experiment_name = 'linear'
elif modeluse == 'dnn':
  experiment_name = 'simple_dnn'
elif modeluse == 'cnn':
  experiment_name = 'simple_cnn'
elif modeluse == 'cpx':
  experiment_name = 'complex_cnn'
else:
  pass


# --------------------------------------
# Pruning

if type_pruning.startswith('AdaNet'):
  pass
else:
  type_prune_index = type_pruning[:3]
  ensemble_pruning = ensemble_pruning_set[type_prune_index]


this_experiment = os.path.join(experiment_name, type_pruning)
if type_pruning.endswith('W'):
  this_experiment += args.adanet_learn_mixture
directory = os.path.join(LOG_DIR, this_experiment)


# --------------------------------------
# Logs

TF_ARCH = 'architecture-{}.json'.format(ADANET_ITERATIONS - 1)
TF_SRCP = os.path.join(LOG_DIR, this_experiment)
TF_FILE = PyFile()

csv_file = open(LOG_TLE + '.csv', 'w', newline="")
csv_writer = csv.writer(csv_file)


# ======================================
# Auxilliary


def rm_certain_dir(directory):
  if os.path.exists(directory):
    shutil.remove(directory)
  logger.warn("remove_previous_model: {:s}/{:s}".format(
      LOG_DIR, this_experiment))


# --------------------------------------
# Log at the beginning


def log_aux_begins():
  logger.warn("TF_LOG_TLE:  {}".format(TF_LOG_TLE))
  logger.warn("LOG_TLE   :  {}".format(LOG_TLE))
  logger.warn("LOG_DIR   :  {}".format(LOG_DIR))
  logger.warning("")

  logger.warn("Data Set   : {}".format(args.dataset))
  logger.warn("Model Setup: {}".format(args.model_setting))
  logger.warning("")

  logger.warn("type_pruning = {}".format(type_pruning))
  logger.warn("thinp_alpha  = {}".format(thinp_alpha))
  logger.warn("cuda_device to use GPU  = {}".format(args.cuda_device))
  logger.warn("if successful using gpu = {}".format(
      tf.test.is_gpu_available()))
  logger.warning("")

  logger.warn("RANDOM_SEED   = {}".format(RANDOM_SEED))
  logger.warn("LEARNING_RATE = {}".format(LEARNING_RATE))
  logger.warn("TRAIN_STEPS   = {}".format(TRAIN_STEPS))
  logger.warn("BATCH_SIZE    = {}".format(BATCH_SIZE))

  logger.warn("ADANET_ITERATIONS     = {}".format(ADANET_ITERATIONS))
  logger.warn("ADANET_LAMBDA         = {}".format(ADANET_LAMBDA))
  logger.warn("LEARN_MIXTURE_WEIGHTS = {}".format(LEARN_MIXTURE_WEIGHTS))
  logger.warning("\n\n\n")


# --------------------------------------
# Log in the middle


def log_aux_middle(results, ensem_arch, time_elapsed):
  logger.info("")
  logger.info("Accuracy: {}".format(results["accuracy"]))
  logger.info("Loss    : {}".format(results["average_loss"]))

  logger.info("{:17s} completed in {} min".format(
      experiment_name, time_elapsed))
  logger.info("ensemble_architecture: {}".format(ensem_arch))

  logger.warn("`cuda_device to use GPU = {}".format(
      args.cuda_device))
  logger.warn("If successful using gpu = {}".format(
      tf.test.is_gpu_available()))
  logger.warning("\n\n\n")


# --------------------------------------
# Log at the end


def log_aux_ending():
  TF_ARCH = TF_FILE.find_architecture(TF_ARCH, TF_SRCP, logger)
  TF_FILE.copy_architecture(TF_ARCH, TF_SRCP, './',
                            TF_LOG_TLE + '-', logger)
  TF_DSTN = TF_LOG_TLE + '-' + TF_ARCH
  TF_DICT = TF_FILE.read_architecture(TF_DSTN)

  logger.info("")
  logger.info("{}- {}".format(TF_LOG_TLE, TF_ARCH))
  logger.info("\tensemble_candidate_name: {}".format(TF_DICT["ensemble_candidate_name"]))
  logger.info("\tensembler_name  :      : {}".format(TF_DICT["ensembler_name"]))
  logger.info("\tglobal_step     :      : {}".format(TF_DICT["global_step"]))
  logger.info("\titeration_number:      : {}".format(TF_DICT["iteration_number"]))
  logger.info("\treplay_indices  :      : {}".format(TF_DICT["replay_indices"]))
  subnetworks = TF_DICT["subnetworks"]
  number_ofit = len(subnetworks)
  number_temp = []  # iteration_number
  logger.info("\t`number of` subnetworks: {}".format(number_ofit))
  for k in range(number_ofit):
    logger.info("\t\titeration_number={:2d}  builder_name= {}".format(
        subnetworks[k]["iteration_number"], subnetworks[k]["builder_name"]))
    number_temp.append(subnetworks[k]["iteration_number"])
  logger.info("\t`No. iteration` subnets: {}".format(number_temp))
  logger.info("")
  logger.info("-----------")

  avg_loss = results["average_loss"]
  accuracy = results["accuracy"]
  diversity = ''
  csv_writer.writerow([
      experiment_name, os.path.split(this_experiment)[-1],
      avg_loss, accuracy, time_elapsed, number_ofit, accuracy * 100.,
      "{}".format(number_temp), ensem_arch,
      TF_DICT["ensemble_candidate_name"], TF_DICT["ensembler_name"],
      TF_DICT["global_step"], TF_DICT["iteration_number"],
      "{}".format(TF_DICT["replay_indices"])])


# ======================================
# Experiment


# --------------------------------------
# Pruning


if type_pruning.startswith('AdaNetO'):
  creator = AdaNetOriginal(RANDOM_SEED)
elif type_pruning.startswith('AdaNetW'):
  creator = AdaNetVariants(RANDOM_SEED)
elif type_pruning.endswith('O'):
  creator = AdaPruOriginal(RANDOM_SEED, type_pruning)
elif type_pruning.endswith('W'):
  creator = AdaNetVariants(RANDOM_SEED, type_pruning)
else:
  raise ValueError("invalid `type_pruning`.")


creator.assign_expt_params(NUM_CLASS, this_experiment, LOG_DIR)
creator.assign_train_param(LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS)
creator.assign_adanet_para(
    ADANET_ITERATIONS, ADANET_LAMBDA, LEARN_MIXTURE_WEIGHTS)


# --------------------------------------


# --------------------------------------
# Train and evaluation


# --------------------------------------


# --------------------------------------


# ======================================
# Cross validation


# --------------------------------------
# Single execution

if nb_cv <= 1:
  input_fn = super_input_fn(
      X_train, y_train, X_test, y_test, NUM_SHAPE, RANDOM_SEED)
  head, feature_columns = establish_baselines(
      NUM_CLASS, NUM_SHAPE, FEATURES_KEY)

  if os.path.exists(LOG_TLE + '.log'):
    os.remove(LOG_TLE + '.log')
  log_file = logging.FileHandler(LOG_TLE + '.log')
  log_file.setLevel(logging.DEBUG)
  log_file.setFormatter(formatter)
  logger.addHandler(log_file)

  log_aux_begins()
  rm_certain_dir(directory)

  since = time.time()
  logger.warning("experiment_name: {}".format(experiment_name))
  logger.warning("this_experiment: {}".format(this_experiment))

  creator.assign_SAEP_adapru(ensemble_pruning, thinp_alpha, logger)
  estimator = creator.create_estimator(
      modeluse, feature_columns, head, input_fn)
  results = creator.train_and_evaluate(estimator, input_fn)

  ensem_arch = creator.ensemble_architecture(results)
  time_elapsed = time.time() - since
  time_elapsed /= 60.  # minutes

  log_aux_middle(results, ensem_arch, time_elapsed)
  log_aux_ending()

  csv_file.close()
  sys.exit()


# --------------------------------------
# Dataset

X_dataset = np.concatenate([X_train, X_test], axis=0)
y_dataset = np.concatenate([y_train, y_test], axis=0)
del X_train, y_train, X_test, y_test

split_idx = situation_cross_validation(nb_cv, y_dataset)


# --------------------------------------
# Cross validation

for i in range(nb_cv):
  idx_trn, idx_tst = split_idx[i]

  X_trn = X_dataset[idx_trn]
  y_trn = y_dataset[idx_trn]
  X_tst = X_dataset[idx_tst]
  y_tst = y_dataset[idx_tst]
  del idx_trn, idx_tst

  input_fn = super_input_fn(
      X_trn, y_trn, X_tst, y_tst, NUM_SHAPE, RANDOM_SEED)
  head, feature_columns = establish_baselines(
      NUM_CLASS, NUM_SHAPE, FEATURES_KEY)

  wr_cv = "_cv" + str(i + 1)
  if os.path.exist(LOG_TLE + wr_cv + '.log'):
    os.remove(LOG_TLE + wr_cv + '.log')
  log_file = logging.FileHandler(LOG_TLE + wr_cv + '.log')
  log_file.setLevel(logging.DEBUG)
  log_file.setFormatter(formatter)
  logger.addHandler(log_file)

  log_aux_begins()
  rm_certain_dir(directory)
  since = time.time()

  creator.assign_SAEP_adapru(ensemble_pruning, thinp_alpha, logger)
  estimator = creator.create_estimator(
      modeluse, feature_columns, head, input_fn)
  results = creator.train_and_evaluate(estimator, input_fn)

  ensem_arch = creator.ensemble_architecture(results)
  time_elapsed = time.time() - since
  time_elapsed /= 60.  # minutes

  log_aux_middle(results, ensem_arch, time_elapsed)
  log_aux_ending()


csv_file.close()
del csv_writer
