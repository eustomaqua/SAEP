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
from execute import (utilise_SAEP, auxrun_expts,
                     run_experiment, BK_LOG_LEV)
from classes import PyFile


# ======================================
# Hyper-parameters


# --------------------------------------
# Argparser

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=BK_LOG_LEV)

args = default_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
TF_LOG_TLE, LOG_TLE, LOG_DIR = default_logs(args)


# --------------------------------------
# Parameters

RANDOM_SEED = None
if args.random_seed != 'None':
  RANDOM_SEED = int(args.random_seed)

LEARN_MIXTURE_WEIGHTS = False
if args.adanet_learn_mixture == 'T':
  LEARN_MIXTURE_WEIGHTS = True


# --------------------------------------
# Pruning

thinp_alpha = args.thinp_alpha
type_pruning = args.type_pruning


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

if type_pruning[:-2] not in ['AdaNet', 'PRS', 'PAP', 'PIE']:
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
tflog = logging.getLogger('tensorflow')
if os.path.exists(TF_LOG_TLE + '.txt'):
  os.remove(TF_LOG_TLE + '.txt')
tf_fh = logging.FileHandler(TF_LOG_TLE + '.txt')
# tf_fh.setLevel(logging.DEBUG)
tf_fh.setLevel(BK_LOG_LEV)
tf_fm = logging.Formatter(logging.BASIC_FORMAT, None)
tf_fh.setFormatter(tf_fm)
tflog.addHandler(tf_fh)

TF_ARCH = 'architecture-{}.json'.format(ADANET_ITERATIONS - 1)
TF_SRCP = os.path.join(LOG_DIR, this_experiment)
TF_FILE = PyFile()

csv_file = open(LOG_TLE + '.csv', 'w', newline="")
csv_writer = csv.writer(csv_file)

creator.assign_SAEP_logger(logger)


# --------------------------------------
# Auxilliary


def output_starts(logger):
  logger.info("TF_LOG_TLE:  {}".format(TF_LOG_TLE))
  logger.info("   LOG_TLE:  {}".format(LOG_TLE))
  logger.info("   LOG_DIR:  {}".format(LOG_DIR))
  logger.info("")

  logger.info("Model Setup:  {}".format(args.model_setting))
  logger.info("Data Set   :  {}".format(args.dataset))
  logger.info("Multiclass?:  {}".format(not args.binary))
  if args.binary:
    logger.info("Binary pair:  {} -/- {}".format(
        args.label_zero, args.label_one))
  logger.info("")

  logger.info("cuda_device to use GPU = {}".format(args.cuda_device))
  logger.info("whether usage succeeds = {}".format(
      tf.test.is_gpu_available()))
  logger.info("type_pruning         = {}".format(type_pruning))
  if type_pruning.startswith('PIE'):
    logger.info("thinp_alpha (alpha)  = {}".format(thinp_alpha))
  if type_pruning.endswith('W'):
    logger.info(
        "LEARN_MIXTURE_WEIGHTS= {}".format(LEARN_MIXTURE_WEIGHTS))
  logger.info("")

  logger.info("RANDOM_SEED       = {}".format(RANDOM_SEED))
  logger.info("LEARNING_RATE     = {}".format(LEARNING_RATE))
  logger.info("TRAIN_STEPS       = {}".format(TRAIN_STEPS))
  logger.info("BATCH_SIZE        = {}".format(BATCH_SIZE))
  logger.info("ADANET_ITERATIONS = {}".format(ADANET_ITERATIONS))
  logger.info("ADANET_LAMBDA     = {}".format(ADANET_LAMBDA))
  logger.info("")

  if os.path.exists(directory):
    shutil.rmtree(directory)
    logger.info("remove_previous_model: {:s}/{:s}"
                "".format(LOG_DIR, this_experiment))
  logger.info("experiment_name: {}".format(experiment_name))
  logger.info("this_experiment: {}".format(this_experiment))
  # logger.info("-----------\n\n")
  logger.info("-----------\n")


def output_ending(logger, since, wr_cv='_sg'):
  time_elapsed = time.time() - since
  # time_elapsed /= 60.  # minutes

  logger.info("")
  logger.info("{:17s}".format(experiment_name))
  logger.info("{:17s} starts at {:s}".format('', time.strftime(
      "%d-%b-%Y %H:%M:%S", time.localtime(since))))
  logger.info("{:17s} finish at {:s}".format('', time.strftime(
      "%d-%b-%Y %H:%M:%S", time.localtime(time.time()))))
  logger.info("{:17s} completed at {:.0f}m {:.2f}s".format(
      '', time_elapsed // 60, time_elapsed % 60))
  logger.info("The entire duration is: {:.6f} min".format(
      time_elapsed / 60))

  '''
  logger.info("Saved location:")
  logger.info("\tLOG_DIR: {:s}".format(LOG_DIR))
  logger.info("\tLOG_TLE: {:s}".format(LOG_TLE + wr_cv))
  logger.info("")
  logger.info("`cuda_device to use GPU =  {}".format(
      args.cuda_device))
  logger.info("if successful using gpu =  {}".format(
      tf.test.is_gpu_available()))
  '''
  logger.info("-----------\n")

  csv_temp = time_elapsed / 60.  # minutes
  return csv_temp


def output_arches(logger, wr_cv='_sg',
                  csv_temp=(), csv_rows=(), arch=TF_ARCH):
  TF_ARCH = TF_FILE.find_architecture(arch, TF_SRCP, logger)

  # TF_ARCH = TF_FILE.find_architecture(TF_ARCH, TF_SRCP, logger)
  TF_FILE.copy_architecture(TF_ARCH, TF_SRCP, './',
                            TF_LOG_TLE + wr_cv + '-', logger)
  TF_DSTN = TF_LOG_TLE + wr_cv + '-' + TF_ARCH
  TF_DICT = TF_FILE.read_architecture(TF_DSTN)

  logger.info("")
  # logger.info("{}- {}".format(TF_LOG_TLE, TF_ARCH))
  logger.info("{} {}".format(TF_LOG_TLE, TF_ARCH))

  logger.info("\tensemble_candidate_name: {}"
              "".format(TF_DICT["ensemble_candidate_name"]))
  logger.info("\tensembler_name  :      : {}"
              "".format(TF_DICT["ensembler_name"]))
  logger.info("\tglobal_step     :      : {}"
              "".format(TF_DICT["global_step"]))
  logger.info("\titeration_number:      : {}"
              "".format(TF_DICT["iteration_number"]))
  logger.info("\treplay_indices  :      : {}"
              "".format(TF_DICT["replay_indices"]))

  subnetworks = TF_DICT["subnetworks"]
  number_ofit = len(subnetworks)
  number_temp = []  # iteration_number
  logger.info("\t`number of` subnetworks: {}".format(number_ofit))
  for k in range(number_ofit):
    logger.info("\t\titeration_number={:2d}  builder_name= {}".format(
        subnetworks[k]["iteration_number"],
        subnetworks[k]["builder_name"]))
    number_temp.append(subnetworks[k]["iteration_number"])
  logger.info("\t`No. iteration` subnets: {}".format(number_temp))
  logger.info("")
  logger.info("-----------")

  csv_writer.writerow([
      'experiment_name', 'type_pruning', 'wr_cv',
      'average_loss', 'accuracy (%)',
      'diver_weight', 'diver_subnet',  # with or without weights
      'time_cost (min)', 'space_cost (size)',
      'subnets', 'replay_indices'
  ])

  csv_temp = [
      experiment_name,
      os.path.split(this_experiment)[-1],
      wr_cv
  ] + csv_temp + [
      number_ofit,
      "{}".format(number_temp),

      # TF_DICT["ensemble_candidate_name"],
      # TF_DICT["ensembler_name"],
      # TF_DICT["global_step"],
      # TF_DICT["iteration_number"],

      "{}".format(TF_DICT["replay_indices"])
  ]

  csv_writer.writerow(csv_temp)
  csv_writer.writerow(csv_rows[0])
  for ijk in csv_rows[1:]:
    csv_writer.writerow(['', '', ''] + ijk)
  # csv_writer.writerows(csv_rows)

  """
  wrcv_csvfile = open(LOG_TLE + wr_cv + '.csv', 'w', newline="")
  wrcv_csvwrit = csv.writer(wrcv_csvfile)
  # wrcv_csvwrit.writerow(csv_rows[0])
  # for _, _, _, i, j, k in csv_rows[1:]:
  #   wrcv_csvwrit.writerow([i, j, k])
  wrcv_csvwrit.writerows(csv_rows)

  wrcv_csvfile.close()
  del wrcv_csvwrit
  """


# --------------------------------------
# Single execution


nb_cv = args.cross_validation

if nb_cv <= 1:
  # wr_cv = '_sing'
  # wr_cv = ''
  wr_cv = '_sg'

  run_experiment(X_train, y_train, X_test, y_test,
                 NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                 LOG_TLE, wr_cv, logger, formatter,
                 creator, modeluse,
                 output_starts, output_ending, output_arches)

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
                 LOG_TLE, wr_cv, logger, formatter,
                 creator, modeluse,
                 output_starts, output_ending, output_arches)


# -----------------------------------------


logger.info("")
discard = os.path.join(os.getcwd(), "*.json")
for fname in glob.glob(discard):
  os.remove(fname)
  logger.info("Deleted " + str(fname))

# discard = glob.glob("*.txt")
# discard.remove("requirements.txt")
# for fname in discard:
#   os.remove(fname)
#   logger.info("Deleted " + fname)

csv_file.close()
del csv_writer


# if __name__ == "__main__":
#   pass
#   # output_starts(logger)
#   # since = time.time()
#   # output_ending(logger, since)
#   # output_arches(logger, '_sg')


# python main.py -cv 1
# python main.py -cv 2 -bi
# python.main.py
