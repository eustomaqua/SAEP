# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import json
import os
import shutil

import re
import tensorflow.compat.v1 as tf
import adanet

from saep.subnetwork_generator import Subnetwork as PrunedSubSubnetwork
from saep.subnetwork_generator import Generator as PrunedSubGenerator
from saep.subnetwork_generator import Builder as PrunedSubBuilder


# --------------------------------------


_NUM_LAYERS_KEY = "num_layers"


class SimpleCNNBuilder(adanet.subnetwork.Builder):
  """Builds a CNN subnetwork for AdaNet."""

  def __init__(self, learning_rate, max_iteration_steps, seed):
    """Initializes a `SimpleCNNBuilder`.
    Args:
        learning_rate: The float learning rate to use.
        max_iteration_steps: The number of steps per iteration.
        seed: The random seed.
    Returns:
        An instance of `SimpleCNNBuilder`.
    """
    self._learning_rate = learning_rate
    self._max_iteration_steps = max_iteration_steps
    self._seed = seed

  def build_subnetwork(self, features, logits_dimension, training,
                       iteration_step, summary, previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""
    images = list(features.values())[0]
    # Visualize some of the input images in TensorBoard.
    summary.image("images", images)

    # input : [?, hw,hw, 3]  # 32 or 28
    # output: [?, hw,hw,16] -> [?, hw/2,hw/2,16] -> [?, hw**2/4 *16=4*hw**2]
    kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                               padding="same", activation="relu",
                               kernel_initializer=kernel_initializer)(images)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=64, activation="relu",
                              kernel_initializer=kernel_initializer)(x)
    # output: [?, 64] -> [?, 10]

    # The `Head` passed to adanet.Estimator will apply the softmax activation.
    logits = tf.keras.layers.Dense(units=10, activation=None,
                                   kernel_initializer=kernel_initializer)(x)
    # Use a constant complexity measure, since all subnetworks have the same
    # architecture and hyperparameters.

    complexity = tf.constant(1)
    return adanet.Subnetwork(last_layer=x, logits=logits,
                             complexity=complexity, persisted_tensors={})

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels, 
                                iteration_step, summary, previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""
    # Momentum optimizer with cosine learning rate decay works well with CNNs.
    learning_rate = tf.train.cosine_decay(learning_rate=self._learning_rate,
                                          global_step=iteration_step,
                                          decay_steps=self._max_iteration_steps)
    optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
    # NOTE: The `adanet.Estimator` increments the global step.
    return optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""
    return tf.no_op("mixture_weights_train_op")

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""
    return "simple_cnn"  # self._name


class SimpleCNNGenerator(adanet.subnetwork.Generator):
  """Generates a `SimpleCNN` at each iteration."""

  def __init__(self, learning_rate, max_iteration_steps, seed=None):
    """Initializes a `Generator` that builds `SimpleCNNs`.
    Args:
        learning_rate: The float learning rate to use.
        max_iteration_steps: The number of steps per iteration.
        seed: The random seed.
    Returns:
        An instance of `Generator`.
    """
    self._seed = seed
    self._dnn_builder_fn = functools.partial(
        SimpleCNNBuilder,
        learning_rate=learning_rate,
        max_iteration_steps=max_iteration_steps)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""
    seed = self._seed
    # Change the seed according to the iteration so that each subnetwork
    # learns something different.
    if seed is not None:
      seed += iteration_number
    return [self._dnn_builder_fn(seed=seed)]


# ===============================
# Self-defined


class PyFile(object):
  def find_architecture(self, filename, srcpath, logger=None):
    srcfile = os.path.join(srcpath, filename)
    if not os.path.exists(srcfile):
      if logger:
        logger.info("")
        logger.info("     srcpath:  {}".format(srcpath))
        logger.info("No such file:  {}".format(filename))
      else:
        print("\nNo such file:  {}".format(srcfile))

      # ls | grep architecture
      archs = os.listdir(srcpath)
      archs = list(filter(lambda x: 'architecture' in x, archs))
      # archs = list(filter(lambda x: re.search(r'architecture', x), archs))
      archs = sorted(archs)

      nums = [i.split('-')[1] for i in archs]
      nums = [i.split('.')[0] for i in nums]
      nums = sorted(int(i) for i in nums)

      if logger:
        logger.info("Last arch is:  architecture-{}.json".format(nums[-1]))
      else:
        print("Last arch is:  architecture-{}.json".format(nums[-1]))
      return 'architecture-{}.json'.format(nums[-1])

    return filename

  def copy_architecture(self, filename, srcpath, dstpath='./',
                        dstname='', logger=None):
    srcfile = os.path.join(srcpath, filename)
    dstfile = os.path.join(dstpath, dstname + filename)
    if not os.path.exists(srcfile) and logger:
      logger.info("No such file:  {}".format(srcfile))
    elif not os.path.exists(srcfile):
      print("No such file:  {}".format(srcfile))
    shutil.copyfile(srcfile, dstfile)

    if logger:
      logger.info("Copy architecture-?.json")
      logger.info("\tSrc path:  {}".format(srcfile))
      logger.info("\tDst path:  {}".format(dstfile))
    else:
      print("Copy architecture-?.json")
      print("\tSrc path:  {}".format(srcfile))
      print("\tDst path:  {}".format(dstfile))
    return

  def read_architecture(self, filename, dstpath='./'):
    dstfile = os.path.join(dstpath, filename)
    with open(dstfile, "r") as dstload:
      dstdict = json.load(dstload)
    return dstdict


# --------------------------------------


class PrunedCNNBuilder(PrunedSubBuilder):
  def __init__(self, learning_rate, max_iteration_steps, seed,
               learn_mixture_weights=False):
    self._learning_rate = learning_rate
    self._max_iteration_steps = max_iteration_steps
    self._seed = seed
    self._learn_mixture_weights = learn_mixture_weights

  def build_subnetwork(self, features, logits_dimension, training,
                       iteration_step, summary, previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""
    images = list(features.values())[0]
    summary.image("images", images)
    kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                               padding="same", activation="relu",
                               kernel_initializer=kernel_initializer)(images)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=64, activation="relu",
                              kernel_initializer=kernel_initializer)(x)
    logits = tf.keras.layers.Dense(units=10, activation=None,
                                   kernel_initializer=kernel_initializer)(x)
    complexity = tf.constant(1)
    return PrunedSubSubnetwork(last_layer=x, logits=logits,
                               complexity=complexity, persisted_tensors={})

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels, 
                                iteration_step, summary, previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""
    learning_rate = tf.train.cosine_decay(learning_rate=self._learning_rate,
                                          global_step=iteration_step,
                                          decay_steps=self._max_iteration_steps)
    self._optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""
    if not self._learn_mixture_weights:
      return tf.no_op("mixture_weights_train_op")
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""
    return "pruned_cnn"  # self._name


class PrunedCNNGenerator(PrunedSubGenerator):
  def __init__(self, learning_rate, max_iteration_steps, seed=None, learn_mixture_weights=False):
    self._seed = seed
    self._dnn_builder_fn = functools.partial(
        PrunedCNNBuilder, learning_rate=learning_rate,
        max_iteration_steps=max_iteration_steps,
        learn_mixture_weights=learn_mixture_weights)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    seed = self._seed
    if seed is not None:
      seed += iteration_number
    return [self._dnn_builder_fn(seed=seed)]


# --------------------------------------


class ComplexCNNBuilder(PrunedSubBuilder):
  def __init__(self, learning_rate, max_iteration_steps, seed, learn_mixture_weights=False):
    self._learning_rate = learning_rate
    self._max_iteration_steps = max_iteration_steps
    self._seed = seed
    self._learn_mixture_weights = learn_mixture_weights

  def build_subnetwork(self, features, logits_dimension, training,
                       iteration_step, summary, previous_ensemble=None):
    images = list(features.values())[0]
    summary.image("images", images)
    kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                               padding="same", activation="relu",
                               kernel_initializer=kernel_initializer)(images)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    # input : [?, hw, hw,  3]  # 32 or 28
    # output: [?, hw, hw, 16]  # [?, hw/2, hw/2, 16]

    x = tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/2, .., 64]
    x = tf.keras.layers.Conv2D(
        filters=256, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/2, .., 128]
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/2, .., 256]
    x = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2)(x)  # [?, hw/4, .., 256] i.e., 16*hw**2

    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/4, .., 128]
    x = tf.keras.layers.Conv2D(
        filters=1024, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/4, .., 256]
    x = tf.keras.layers.Conv2D(
        filters=2048, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/4, .., 512]
    x = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2)(x)  # [?, hw/4, .., 512] i.e., 32

    x = tf.keras.layers.Conv2D(
        filters=2048, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/4, .., 512]
    x = tf.keras.layers.Conv2D(
        filters=4096, kernel_size=3, padding="same", activation="relu",
        kernel_initializer=kernel_initializer)(x)  # [?, hw/4, .., 512]
    x = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2)(x)  # [?, hw/8, .., 1024]

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation="relu",
                              kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Dense(units=512, activation="relu",
                              kernel_initializer=kernel_initializer)(x)
    # output: [?, hw**2/16 *64 = 4*hw**2] -> [?, 64]

    logits = tf.keras.layers.Dense(units=logits_dimension, activation=None,
                                   kernel_initializer=kernel_initializer)(x)
    complexity = tf.constant(1)
    return PrunedSubSubnetwork(last_layer=x, logits=logits,
                               complexity=complexity,
                               persisted_tensors={})

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary,
                                previous_ensemble=None):
    learning_rate = tf.train.cosine_decay(
        learning_rate=self._learning_rate,
        global_step=iteration_step,
        decay_steps=self._max_iteration_steps)
    self._optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    if not self._learn_mixture_weights:
      return tf.no_op("mixture_weights_train_op")
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  @property
  def name(self):
    return "pruned_cpx"  # complex cnn


class ComplexCNNGenerator(PrunedSubGenerator):
  def __init__(self, learning_rate, max_iteration_steps, seed=None, learn_mixture_weights=False):
    self._seed = seed
    self._dnn_builder_fn = functools.partial(
        ComplexCNNBuilder, learning_rate=learning_rate,
        max_iteration_steps=max_iteration_steps,
        learn_mixture_weights=learn_mixture_weights)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    seed = self._seed
    if seed is not None:
      seed += iteration_number
    return [self._dnn_builder_fn(seed=seed)]
