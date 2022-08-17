# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
from copy import deepcopy
import os

import logging
import numpy as np

from dataset import data_to_feed_in
from classes import PyFile
logging.basicConfig(level=logging.DEBUG)


# ======================================
# Preliminaries


# --------------------------------------
# Argparser


def default_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-data', '--dataset', type=str, default='cifar10',
                      help='Data set')
  parser.add_argument('-model', '--model_setting', type=str, default='dnn',
                      help='Builder')

  parser.add_argument('-type', '--type_pruning', type=str,
                      default='SAEP.O', choices=[
                          'AdaNet.O', 'SAEP.O', 'PRS.O', 'PAP.O', 'PIE.O',
                          'AdaNet.W', 'SAEP.W', 'PRS.W', 'PAP.W', 'PIE.W'],
                      help='How to prune the AdaNet')
  parser.add_argument('-alpha', '--thinp_alpha', type=float, default=0.5,
                      help='The value of alpha in PIE')

  parser.add_argument('-cv', '--cross_validation', type=int, default=5,
                      help='Cross validation')
  parser.add_argument('-bi', '--binary', action='store_true',
                      help='binary classification (pairs)')
  parser.add_argument('-c0', '--label_zero', type=int, default=4,
                      help='the first label in pairs')
  parser.add_argument('-c1', '--label_one', type=int, default=9,
                      help='the second label in pairs')

  parser.add_argument('-device', '--cuda_device', type=str, default='0',
                      help='cuda_visible_devices')

  parser.add_argument('-lr', '--learning_rate', type=float, default=0.003,
                      help='LEARNING_RATE')
  parser.add_argument('-ts', '--train_steps', type=int, default=5000,
                      help='TRAIN_STEPS')
  parser.add_argument('-bs', '--batch_size', type=int, default=64,
                      help='BATCH_SIZE')
  parser.add_argument('-rs', '--random_seed', type=str, default='None',
                      help='RANDOM_SEED')

  parser.add_argument('-it', '--adanet_iterations', type=int, default=2,
                      help='ADANET_ITERATIONS')
  parser.add_argument('-mix', '--adanet_learn_mixture', type=str,
                      default='F', choices=['T', 'F'],
                      help='LEARN_MIXTURE_WEIGHTS')
  parser.add_argument('-lam', '--adanet_lambda', type=float, default=0,
                      help='ADANET_LAMBDA')

  # return parser
  args = parser.parse_args()
  return args


# --------------------------------------
# Logs


def default_logs(args, saved='tmpmodels'):
  LOG_TLE = args.dataset
  LOG_TLE += '_cv' + str(args.cross_validation)
  LOG_TLE += '_it' + str(args.adanet_iterations)
  LOG_TLE += '_lr' + str(args.learning_rate)
  LOG_TLE += '_bs' + str(args.batch_size)
  LOG_TLE += '_ts' + str(args.train_steps // 1000)

  LOG_DIR = os.path.join(os.getcwd(), saved)
  LOG_DIR = os.path.join(LOG_DIR, args.dataset)
  TF_LOG_TLE = args.dataset

  if args.binary:
    feat_pair = (args.label_zero, args.label_one)
    feat_temp = 'pair' + ''.join(map(str, feat_pair))
  else:
    feat_temp = 'multi'

  LOG_DIR = os.path.join(LOG_DIR, feat_temp)
  LOG_TLE += '_' + feat_temp
  TF_LOG_TLE += '_' + feat_temp
  LOG_TLE += '_' + args.model_setting
  TF_LOG_TLE += '_' + args.model_setting

  thinp_alpha = args.thinp_alpha
  type_pruning = args.type_pruning
  LOG_TLE += '_' + type_pruning
  TF_LOG_TLE += '_' + type_pruning
  if type_pruning.startswith('PIE'):
    LOG_TLE += str(thinp_alpha)
    TF_LOG_TLE += str(thinp_alpha)
  if type_pruning.endswith('W'):
    LOG_TLE += str(args.adanet_learn_mixture)
    TF_LOG_TLE += str(args.adanet_learn_mixture)

  # if args.cross_validation > 0:
  #   LOG_TLE += '_cv' + str(args.cross_validation)
  # else:
  #   LOG_TLE += '_sing'
  # return LOG_TLE, LOG_DIR, feat_temp

  return TF_LOG_TLE, LOG_TLE, LOG_DIR


# def default_recs(args, saved='tmpmodels'):
#   TF_LOG_TLE, LOG_TLE, LOG_DIR = default_logs(args, saved)
def default_recs(TF_LOG_TLE):

  logger = logging.getLogger('test')
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s: %(levelname)s | %(message)s')

  # Logs of TensorFlow
  # TF_LOG_TLE = args.dataset

  tflog = logging.getLogger('tensorflow')
  if os.path.exists(TF_LOG_TLE + '.txt'):
    os.remove(TF_LOG_TLE + '.txt')
  tf_fh = logging.FileHandler(TF_LOG_TLE + '.txt')
  tf_fh.setLevel(logging.DEBUG)
  tf_fm = logging.Formatter(logging.BASIC_FORMAT, None)
  tf_fh.setFormatter(tf_fm)
  tflog.addHandler(tf_fh)

  # nb_iter = args.cross_validation
  # wr_cv = "_cv" + str(i + 1)
  TF_ARCH = 'architecture-'

  return logger, tflog


# --------------------------------------
# Data set


def default_feed(args):
  datafeed = args.dataset
  if not args.binary:
    return data_to_feed_in(datafeed, False)

  c0, c1 = args.label_zero, args.label_one
  return datafeed(datafeed, True, c0, c1)


# ======================================
# Auxilliary


# --------------------------------------
# Cross validation:
# different ways to split data


def situation_cross_validation(nb_iter, y, split_type='cross_valid_v2'):
  if split_type not in ['cross_valid_v3', 'cross_valid_v2',
                        'cross_validation', 'cross_valid']:
    raise ValueError("invalid `split_type`.")

  y = np.array(y)
  vY = np.unique(y)
  dY = len(vY)
  iY = [np.where(y == j)[0] for j in vY]  # indexes
  lY = [len(j) for j in iY]  # length

  tY = [np.copy(j) for j in iY]  # temp_index
  for j in tY:
    np.random.shuffle(j)
  sY = [int(np.floor(j / nb_iter)) for j in lY]  # split length
  if nb_iter == 2:
    sY = [int(np.floor(j / (nb_iter + 1))) for j in lY]
  elif nb_iter == 1:
    sY = [int(np.floor(j / (nb_iter + 1))) for j in lY]

  split_idx = []
  for k in range(1, nb_iter + 1):
    i_tst, i_val, i_trn = [], [], []

    for i in range(dY):
      k_former = sY[i] * (k - 1)
      k_middle = sY[i] * k
      k_latter = sY[i] * (k + 1) if k != nb_iter else sY[i]

      i_tst.append(tY[i][k_former: k_middle])
      if k != nb_iter:
        i_val.append(tY[i][k_middle: k_latter])
        i_trn.append(
            np.concatenate(
                [tY[i][k_latter:], tY[i][: k_former]], axis=0))
      else:
        i_val.append(tY[i][: k_latter])
        i_trn.append(
            np.concatenate(
                [tY[i][k_middle:], tY[i][k_latter: k_former]], axis=0))

    i_tst = np.concatenate(i_tst, axis=0).tolist()
    i_val = np.concatenate(i_val, axis=0).tolist()
    i_trn = np.concatenate(i_trn, axis=0).tolist()

    temp_ = (deepcopy(i_trn), deepcopy(i_val), deepcopy(i_tst))
    if split_type.endswith('v2'):
      temp_ = (deepcopy(i_trn + i_val), deepcopy(i_tst))

    split_idx.append(deepcopy(temp_))
    del k_former, k_middle, k_latter, i_tst, i_val, i_trn
  del k, y, vY, dY, iY, lY, tY, sY, nb_iter
  # gc.collect()
  return deepcopy(split_idx)
