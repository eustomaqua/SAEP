# coding: utf-8
# 5-cross validation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import os
import shutil

import numpy as np
import tensorflow.compat.v1 as tf

from estimat import AdaNetOriginal, AdaNetVariants
from estimat import AdaPruOriginal, AdaPruVariants
from classes import PyFile

from hparam import super_input_fn
from dataset import establish_baselines
from dataset import FEATURES_KEY


BK_LOG_LEV = logging.INFO
# BK_LOG_LEV = logging.DEBUG
logging.basicConfig(level=BK_LOG_LEV)
TF_FILE = PyFile()


# ======================================
# Pruning


ensemble_pruning_set = {
    "AdaNet": "keep_all",
    "SAEP": "keep_all",
    "PRS": "pick_randsear",
    "PAP": "pick_worthful",
    "PIE": "pick_infothin",
}  # not a list anymore

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


def utilise_SAEP(type_pruning='SAEP.O',
                 thinp_alpha=.5,
                 learn_mixture_weights=False,
                 modeluse='dnn',
                 random_seed=None):
  r"""
  Note that SAEP is identical to AdaNet, except for it has more
  information to output for us to analyse.
  """

  if type_pruning.startswith('AdaNet'):
    type_pruning = type_pruning.replace('AdaNet', 'SAEP')

  assert type_pruning[-1] in ['O', 'W']
  assert type_pruning[:-2] in ['SAEP', 'PRS', 'PAP', 'PIE']
  assert modeluse in ['dnn', 'cnn']  # , 'cpx']

  if type_pruning.endswith('O'):
    creator = AdaPruOriginal(random_seed, type_pruning)
  elif not learn_mixture_weights:
    creator = AdaPruVariants(random_seed, type_pruning)
    creator.LEARN_MIXTURE_WEIGHTS = False
  else:
    creator = AdaPruVariants(random_seed, type_pruning)
    creator.LEARN_MIXTURE_WEIGHTS = True

  ens_pruning = ensemble_pruning_set[type_pruning[:-2]]
  if not type_pruning.startswith('PIE'):
    creator.assign_SAEP_adapru(ens_pruning)
  else:
    creator.assign_SAEP_adapru(ens_pruning, thinp_alpha=thinp_alpha)

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
    this_experiment += str(lmw)[0]
  if type_pruning.startswith('PIE'):
    this_experiment += str(thinp_alpha)

  return experiment_name, this_experiment


def output_starts(logger, args, RANDOM_SEED,
                  TF_LOG_TLE, LOG_TLE, LOG_DIR,
                  experiment_name, this_experiment, directory):
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

  type_pruning = args.type_pruning
  thinp_alpha = args.thinp_alpha
  LEARN_MIXTURE_WEIGHTS = args.adanet_learn_mixture

  logger.info("cuda_device to use GPU = {}".format(args.cuda_device))
  logger.info("whether usage succeeds = {}".format(
      tf.test.is_gpu_available()))
  logger.info("type_pruning           = {}".format(type_pruning))
  if type_pruning.startswith('PIE'):
    logger.info("thinp_alpha (alpha)    = {}".format(thinp_alpha))
  if type_pruning.endswith('W'):
    logger.info(
        "LEARN_MIXTURE_WEIGHTS  = {}".format(LEARN_MIXTURE_WEIGHTS))
  logger.info("")

  LEARNING_RATE = args.learning_rate
  BATCH_SIZE = args.batch_size
  TRAIN_STEPS = args.train_steps
  ADANET_LAMBDA = args.adanet_lambda
  ADANET_ITERATIONS = args.adanet_iterations

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

  logger.info("")
  logger.info("-----------")
  return


def output_ending(logger, since, wr_cv='_sg', experiment_name=''):
  time_elapsed = time.time() - since
  # time_elapsed /= 60.  # minutes

  logger.info("")
  logger.info("{:17s}".format(experiment_name))
  logger.info("{:17s} starts at {:s}".format('', time.strftime(
      "%d-%b-%Y %H:%M:%S", time.localtime(since))))
  logger.info("{:17s} finish at {:s}".format('', time.strftime(
      "%d-%b-%Y %H:%M:%S", time.localtime(time.time()))))
  logger.info("{:17s} completed in {:.0f}m {:.2f}s".format(
      '', time_elapsed // 60, time_elapsed % 60))
  logger.info("The entire duration is: {:.6f} min".format(
      time_elapsed / 60))

  logger.info("")
  logger.info("-----------")
  csv_temp = time_elapsed / 60.  # minutes
  return csv_temp


def output_arches(logger, TF_LOG_TLE, arch, srcp, wr_cv='_sg',
                  csv_temp=(), csv_rows=(), csv_writer=None,
                  experiment_name='', this_experiment=''):
  TF_ARCH, TF_SRCP = arch, srcp
  TF_ARCH = TF_FILE.find_architecture(TF_ARCH, TF_SRCP, logger)

  TF_FILE.copy_architecture(TF_ARCH, TF_SRCP, './',
                            TF_LOG_TLE + wr_cv + '-', logger)
  TF_DSTN = TF_LOG_TLE + wr_cv + '-' + TF_ARCH
  TF_DICT = TF_FILE.read_architecture(TF_DSTN)

  logger.info("")
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
  logger.info("\n\n")

  if not csv_writer:
    return
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
  if not csv_rows:
    return
  csv_writer.writerow(csv_rows[0])
  for ijk in csv_rows[1:]:
    csv_writer.writerow(ijk)

  return


# --------------------------------------


def run_SAEP_experiment(X_trn, y_trn, X_tst, y_tst,
                        NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                        LOG_TLE, wr_cv,
                        logger, formatter, csv_writer,
                        creator, modeluse, arch, srcp,
                        TF_LOG_TLE, experiment_name, this_experiment,
                        LOG_DIR, directory, args):
  input_fn = super_input_fn(
      X_trn, y_trn, X_tst, y_tst, NUM_SHAPE, RANDOM_SEED)
  head, feature_columns = establish_baselines(
      NUM_CLASS, NUM_SHAPE, FEATURES_KEY)

  BK_LOG_TLE = LOG_TLE + wr_cv + '.log'
  if os.path.exists(BK_LOG_TLE):
    os.remove(BK_LOG_TLE)
  log_file = logging.FileHandler(BK_LOG_TLE)
  log_file.setLevel(BK_LOG_LEV)
  log_file.setFormatter(formatter)
  logger.addHandler(log_file)

  # output_starts(logger)
  output_starts(logger, args, RANDOM_SEED,
                TF_LOG_TLE, LOG_TLE, LOG_DIR,
                experiment_name, this_experiment, directory)
  since = time.time()
  # logger.warning("experiment_name: {}".format(experiment_name))
  # logger.warning("this_experiment: {}".format(this_experiment))

  estimator = creator.create_estimator(
      modeluse, feature_columns, head, input_fn)
  results, estimator = creator.train_and_evaluate(
      estimator, input_fn)

  adanet_loss = estimator._adanet_loss
  diver_weight = estimator._diver_weight
  diver_subnet = estimator._diver_subnet

  logger.info("")
  logger.info("Accuracy: {}".format(results["accuracy"]))
  logger.info("Loss: {}".format(results["average_loss"]))

  ensem_arch = creator.ensemble_architecture(results)
  logger.info("ensemble_architecture: {}".format(ensem_arch))
  logger.info(" adanet_loss: {}".format(np.mean(adanet_loss)))
  logger.info("diver_weight: {}".format(np.mean(diver_weight)))
  logger.info("diver_subnet: {}".format(np.mean(diver_subnet)))

  # csv_temp = output_ending(logger, since, wr_cv)
  csv_temp = output_ending(logger, since, wr_cv, experiment_name)

  avg_loss = results["average_loss"]
  accuracy = results["accuracy"]
  # diversity = ''

  csv_temp = [
      avg_loss,
      # accurracy,
      # csv_temp,
      # ensem_arch,

      accuracy * 100.,
      np.mean(diver_weight),
      np.mean(diver_subnet),

      # np.std(adanet_loss),
      # np.std(diver_weight),
      # np.std(diver_subnet),

      csv_temp,
  ]

  csv_rows = [[
      'adanet_loss', 'diver_weight', 'diver_subnet',
      '', 'len={}'.format(len(adanet_loss))]]
  for i, j, k in zip(adanet_loss, diver_weight, diver_subnet):
    csv_rows.append([i, j, k])
  output_arches(
      logger, TF_LOG_TLE, arch, srcp,
      wr_cv, csv_temp, csv_rows, csv_writer,
      experiment_name, this_experiment)


def run_AdaNet_experiment(X_trn, y_trn, X_tst, y_tst,
                          NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                          LOG_TLE, wr_cv,
                          logger, formatter, csv_writer,
                          creator, modeluse, arch, srcp,
                          TF_LOG_TLE, experiment_name, this_experiment,
                          LOG_DIR, directory, args):
  input_fn = super_input_fn(
      X_trn, y_trn, X_tst, y_tst, NUM_SHAPE, RANDOM_SEED)
  head, feature_columns = establish_baselines(
      NUM_CLASS, NUM_SHAPE, FEATURES_KEY)

  BK_LOG_TLE = LOG_TLE + wr_cv + '.log'
  if os.path.exists(BK_LOG_TLE):
    os.remove(BK_LOG_TLE)
  log_file = logging.FileHandler(BK_LOG_TLE)
  log_file.setLevel(BK_LOG_LEV)
  log_file.setFormatter(formatter)
  logger.addHandler(log_file)

  output_starts(logger, args, RANDOM_SEED,
                TF_LOG_TLE, LOG_TLE, LOG_DIR,
                experiment_name, this_experiment, directory)
  since = time.time()

  estimator = creator.create_estimator(
      modeluse, feature_columns, head, input_fn)
  results, estimator = creator.train_and_evaluate(
      estimator, input_fn)

  logger.info("Accuracy: {}".format(results["accuracy"]))
  logger.info("Loss: {}".format(results["average_loss"]))

  ensem_arch = creator.ensemble_architecture(results)
  logger.info("ensemble_architecture: {}".format(ensem_arch))

  csv_temp = output_ending(logger, since, wr_cv, experiment_name)

  avg_loss = results["average_loss"]
  accuracy = results["accuracy"]
  diversity = ''

  csv_temp = [
      avg_loss, accuracy * 100, diversity, diversity, csv_temp]
  output_arches(
      logger, TF_LOG_TLE, arch, srcp, wr_cv, csv_temp,
      csv_writer=csv_writer,
      experiment_name=experiment_name,
      this_experiment=this_experiment)


def run_experiment(X_trn, y_trn, X_tst, y_tst,
                   NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                   LOG_TLE, wr_cv,
                   logger, formatter, csv_writer,
                   creator, modeluse, arch, srcp,
                   TF_LOG_TLE, type_pruning,
                   experiment_name, this_experiment,
                   LOG_DIR, directory, args):
  if type_pruning.startswith('AdaNet'):
    run_AdaNet_experiment(X_trn, y_trn, X_tst, y_tst,
                          NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
                          LOG_TLE, wr_cv,
                          logger, formatter, csv_writer,
                          creator, modeluse, arch, srcp,
                          TF_LOG_TLE,
                          experiment_name, this_experiment,
                          LOG_DIR, directory, args)
  else:
    run_SAEP_experiment(
        X_trn, y_trn, X_tst, y_tst,
        NUM_CLASS, NUM_SHAPE, RANDOM_SEED,
        LOG_TLE, wr_cv,
        logger, formatter, csv_writer,
        creator, modeluse, arch, srcp,
        TF_LOG_TLE,
        experiment_name, this_experiment,
        LOG_DIR, directory, args)
