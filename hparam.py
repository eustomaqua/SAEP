# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
from copy import deepcopy
import os

import numpy as np
import tensorflow.compat.v1 as tf

from dataset import DTY_FLT, DTY_INT
from dataset import generator
from dataset import preprocess_image_28
from dataset import preprocess_image_32


# ======================================
# Preliminaries


# --------------------------------------
# Argparser


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
CONFIG = tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS)


# ======================================
# Preliminaries


# --------------------------------------
# Argparser
# Saved Parameters


def default_args_params():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-data', '--dataset', type=str, default='mnist',
      choices=['cifar10', 'mnist', 'fashion_mnist', 'fmnist'],
      help='Data set')
  parser.add_argument(
      '-model', '--model_setting', type=str, default='dnn',
      # choices=['dnn', 'cnn'], help='Builder')
      choices=["dnn", "cnn", 'linear', 'cpx'], help='Builder')
  parser.add_argument(
      '-type', '--type_pruning', type=str, default='AdaNet.O',
      choices=['AdaNet.O', 'SAEP.O', 'PRS.O', 'PAP.O', 'PIE.O',
               'AdaNet.W', 'SAEP.W', 'PRS.W', 'PAP.W', 'PIE.W'],
      help='How to prune the AdaNet')
  parser.add_argument(
      '-alpha', '--thinp_alpha', type=float, default=0.5,
      help='The value of alpha in PIE')

  parser.add_argument(
      '-cv', '--cross_validation', type=int,
      default=1,  # or 5, 1 means no cross_validation
      help='Cross validation')
  parser.add_argument(
      '-bi', '--binary', action='store_true',
      help='binary classification (pairs)')
  parser.add_argument(
      '-c0', '--label_zero', type=int, default=4,
      help='the first label in pairs')
  parser.add_argument(
      '-c1', '--label_one', type=int, default=9,
      help='the second label in pairs')
  parser.add_argument(
      '-device', '--cuda_device', type=str, default='0',
      help='cuda_visible_devices')

  parser.add_argument(
      '-lr', '--learning_rate', type=float, default=0.003,
      help='LEARNING_RATE')
  parser.add_argument(
      '-ts', '--train_steps', type=int, default=5000,
      help='TRAIN_STEPS')
  parser.add_argument(
      '-bs', '--batch_size', type=int, default=64,
      help='BATCH_SIZE')
  parser.add_argument(
      '-rs', '--random_seed', type=str, default='None',
      help='RANDOM_SEED')

  parser.add_argument(
      '-it', '--adanet_iterations', type=int, default=7,
      help='ADANET_ITERATIONS')
  parser.add_argument(
      '-mix', '--adanet_learn_mixture', action='store_true',
      # type=str, default='F', choices=['T', 'F'],
      help='LEARN_MIXTURE_WEIGHTS')
  parser.add_argument(
      '-lam', '--adanet_lambda', type=float, default=0,
      help='ADANET_LAMBDA')

  return parser


# --------------------------------------
# Logs


def default_logs_folder(args, saved='tmpmodels'):
  LOG_TLE = args.dataset
  LOG_TLE += '_cv' + str(args.cross_validation)
  # LOG_TLE += '_it' + str(args.adanet_iterations)
  # LOG_TLE += '_lr' + str(args.learning_rate)
  # LOG_TLE += '_bs' + str(args.batch_size)
  # LOG_TLE += '_ts' + str(args.train_steps // 1000)

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
    lmw = str(args.adanet_learn_mixture)
    LOG_TLE += lmw[0]
    TF_LOG_TLE += lmw[0]

  # if args.cross_validation > 0:
  #   LOG_TLE += '_cv' + str(args.cross_validation)
  # else:
  #   LOG_TLE += '_sing'
  # return LOG_TLE, LOG_DIR, feat_temp

  return TF_LOG_TLE, LOG_TLE, LOG_DIR


# ======================================
# Auxilliary


# --------------------------------------
# Cross validation:
# different ways to split data


def situation_cross_validation(nb_iter, y, split_type='cross_valid_v2'):
  if split_type not in ['cross_valid_v3', 'cross_valid_v2',
                        'cross_validation', 'cross_valid']:
    raise ValueError("invalid `split_type`, {}.".format(split_type))

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


# --------------------------------------
# 1. download the data


def feed_dataset_all_in(datafeed, binary=False, c0=4, c1=9):
  if datafeed.startswith('cifar10'):
    NUM_CLASS = 10
    NUM_SHAPE = (32, 32, 3)
    TARGETS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
  elif datafeed.endswith('mnist'):
    NUM_CLASS = 10
    NUM_SHAPE = (28, 28, 1)
    TARGETS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", ]
  else:
    raise ValueError("No such dataset named {}!".format(datafeed))

  if datafeed == 'cifar10':
    (X_train, y_train), (
        X_test, y_test) = tf.keras.datasets.cifar10.load_data()
  elif datafeed == 'mnist':
    (X_train, y_train), (
        X_test, y_test) = tf.keras.datasets.mnist.load_data()
  elif datafeed == 'fmnist' or datafeed == 'fashion_mnist':
    (X_train, y_train), (
        X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
  else:
    X_train = y_train = X_test = y_test = None

  if datafeed.startswith('cifar'):
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
  if datafeed.endswith('mnist'):
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

  if binary:
    mask_train = (y_train == c0) | (y_train == c1)
    mask_test = (y_test == c0) | (y_test == c1)

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

  return NUM_CLASS, NUM_SHAPE, TARGETS, X_train, y_train, X_test, y_test


# --------------------------------------
# Data Set


def default_data_feedin(args):
  datafeed = args.dataset
  if not args.binary:
    return feed_dataset_all_in(datafeed, False)

  # binary = args.binary
  c0, c1 = args.label_zero, args.label_one
  return feed_dataset_all_in(datafeed, True, c0, c1)


# --------------------------------------
# 1. download the data


def super_input_fn(X_train, y_train, X_test, y_test,
                   NUM_SHAPE, RANDOM_SEED):
  def input_fn(partition, training, batch_size):
    # Generate an input_fn for the Estimator.
    def _input_fn():

      if partition == "train":
        dataset = tf.data.Dataset.from_generator(generator(
            X_train, y_train), (DTY_FLT, DTY_INT), (NUM_SHAPE, ()))
      elif partition == "predict":
        dataset = tf.data.Dataset.from_generator(generator(
            X_test[:30], y_test[:30]), (DTY_FLT, DTY_INT), (NUM_SHAPE, ()))
      else:
        dataset = tf.data.Dataset.from_generator(generator(
            X_test, y_test), (DTY_FLT, DTY_INT), (NUM_SHAPE, ()))

      # We call repeat after shuffling, rather than before, to prevent
      # separate epochs from blending together.
      if training:
        dataset = dataset.shuffle(10 * batch_size,
                                  seed=RANDOM_SEED).repeat()

      if NUM_SHAPE[0] == 32:  # NUM_SHAPE == (32, 32, 3):
        dataset = dataset.map(preprocess_image_32)
      elif NUM_SHAPE[0] == 28:  # NUM_SHAPE == (28, 28, 1):
        dataset = dataset.map(preprocess_image_28)
      else:
        raise ValueError("invalid `NUM_SHAPE`, {}.".format(NUM_SHAPE))

      dataset = dataset.shuffle(buffer_size=10000)
      dataset = dataset.batch(batch_size)
      dataset = dataset.repeat(10)  # num_epochs)

      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    return _input_fn
  return input_fn
