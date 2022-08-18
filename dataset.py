# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow.compat.v1 as tf


# ======================================
# Preliminaries

# --------------------------------------
# Packages


# gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
CONFIG = tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS)


# --------------------------------------
# Hyper-Parameters


DTY_FLT = 'float32'
DTY_INT = 'int32'


# 2. supply the data in TensorFlow
FEATURES_KEY = "images"


# --------------------------------------
# 2. supply the data in TensorFlow


def generator(images, labels):
  """Returns a generator that returns image-label pairs."""
  def _gen():
    for image, label in zip(images, labels):
      yield image, label
  return _gen


def preprocess_image_28(image, label):
  """Preprocesses an image for an `Estimator`."""
  # first lets scale the pixel values to be between 0 and 1.
  image = image / 255.
  # next we reshape the image so that we can apply a 2D convolution to it.
  image = tf.reshape(image, [28, 28, 1])
  # finally the features need to be supplied as a dictionary.
  features = {FEATURES_KEY: image}
  return features, label


def preprocess_image_32(image, label):
  """Preprocesses an image for an `Estimator`."""
  # first lets scale the pixel values to be between 0 and 1.
  image = image / 255.
  # next we reshape the image so that we can apply a 2D convolution to it.
  image = tf.reshape(image, [32, 32, 3])
  # finally the features need to be supplied as a dictionary.
  features = {FEATURES_KEY: image}
  return features, label


def super_input_fn(X_train, y_train, X_test, y_test,
                   NUM_SHAPE, RANDOM_SEED):
  def input_fn(partition, training, batch_size):
    # Generate an input_fn for the Estimator.
    def _input_fn():

      if partition == "train":
        dataset = tf.data.Dataset.from_generator(
            generator(X_train, y_train), (DTY_FLT, DTY_INT), (NUM_SHAPE, ()))
      elif partition == "predict":
        dataset = tf.data.Dataset.from_generator(generator(
            X_test[:30], y_test[:30]), (DTY_FLT, DTY_INT), (NUM_SHAPE, ()))
      else:
        dataset = tf.data.Dataset.from_generator(
            generator(X_test, y_test), (DTY_FLT, DTY_INT), (NUM_SHAPE, ()))

      # We call repeat after shuffling, rather than before, to prevent
      # separate epochs from blending together.
      if training:
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

      if NUM_SHAPE[0] == 32:
        dataset = dataset.map(preprocess_image_32).batch(batch_size)
      elif NUM_SHAPE[0] == 28:
        dataset = dataset.map(preprocess_image_28).batch(batch_size)
      else:
        raise ValueError("invalid `NUM_SHAPE`.")

      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

    return _input_fn
  return input_fn


# --------------------------------------
# 3. launch TensorBoard


# --------------------------------------
# 4. establish baselines


def establish_baselines(NUM_CLASS, NUM_SHAPE, FEATURES_KEY):
  # A `Head` instance defines the loss function and metrics for `Estimators`.
  head = tf.estimator.MultiClassHead(NUM_CLASS)

  # if NUM_CLASS > 2:
  #   head = tf.estimator.MultiClassHead(NUM_CLASS)
  # else:
  #   head = tf.estimator.BinaryClassHead(NUM_CLASS)

  # Some `Estimators` use feature columns to understand their input features.
  feature_columns = [
      tf.feature_column.numeric_column(FEATURES_KEY, shape=NUM_SHAPE)
  ]
  return head, feature_columns


def make_config(experiment_name, RANDOM_SEED, LOG_DIR):
  # Estimator configuration.
  return tf.estimator.RunConfig(
      save_checkpoints_steps=1000,
      save_summary_steps=1000,
      tf_random_seed=RANDOM_SEED,
      model_dir=os.path.join(LOG_DIR, experiment_name))
  # session_config = CONFIG


# --------------------------------------
# 1. download the data


def data_to_feed_in(fed_data, binary=False, c0=4, c1=9):
  # fed_data = args.dataset

  if fed_data.startswith('cifar10'):
    # NUM_CLASS = 10
    NUM_SHAPE = (32, 32, 3)
    TARGETS = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
  elif fed_data.endswith('mnist'):
    # NUM_CLASS = 10
    NUM_SHAPE = (28, 28, 1)
    TARGETS = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
    ]
  else:
    raise ValueError("No such dataset named {}!".format(fed_data))

  if fed_data == 'cifar10':
    (X_train, y_train), (
        X_test, y_test) = tf.keras.datasets.cifar10.load_data()
  elif fed_data == 'mnist':
    (X_train, y_train), (
        X_test, y_test) = tf.keras.datasets.mnist.load_data()
  elif fed_data == 'fmnist' or fed_data == 'fashion_mnist':
    (X_train, y_train), (
        X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
  else:
    X_train = y_train = X_test = y_test = None

  if fed_data.startswith('cifar'):
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
  if fed_data.endswith('mnist'):
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

  if binary:
    mask_train = (y_train == c0) | (y_train == c1)
    mask_test = (y_test == c0) | (y_test == c1)

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

  return NUM_SHAPE, TARGETS, X_train, y_train, X_test, y_test


# ======================================
# Auxilliary


def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""
  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]


def super_obtain_results(input_fn, BATCH_SIZE, TRAIN_STEPS, LOG_DIR,
                         LOG_TLE, logger=None):
  def obtain_results(estimator, experiment_name, since):
    results, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn("train", training=True, batch_size=BATCH_SIZE),
            max_steps=TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn("test", training=False, batch_size=BATCH_SIZE),
            steps=None,
            start_delay_secs=1,
            throttle_secs=1))

    time_elapsed = time.time() - since
    time_elapsed /= 60.  # minutes

    if logger:
      logger.critical("\n\n\n")
      logger.critical("Loss:     {}".format(results["average_loss"]))
      logger.critical("Accuracy: {}".format(results["accuracy"]))
      logger.critical("{:17s} completed in {} min".format(time_elapsed))
    else:
      print("\n\n\n")
      print("Loss:     {}".format(results["average_loss"]))
      print("Accuracy: {}".format(results["accuracy"]))
      print("{:17s} completed in {} min".format(time_elapsed))

    return results['average_loss'], results['accuracy'], time_elapsed
  return obtain_results
