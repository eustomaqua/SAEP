# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow.compat.v1 as tf


# =====================================
# Preliminaries

# -------------------------------------
# Packages


# gc.enable()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
CONFIG = tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS)


# -------------------------------------
# Hyper-Parameters


DTY_FLT = 'float32'
DTY_INT = 'int32'

# 2. supply the data in TensorFlow
FEATURES_KEY = "images"


# -------------------------------------
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


# -------------------------------------
# 3. launch TensorBoard


# -------------------------------------
# 4. establish baselines


def establish_baselines(NUM_CLASS, NUM_SHAPE, FEATURES_KEY):
  # A `Head` instance defines the loss function and metrics for `Estimators`.
  head = tf.estimator.MultiClassHead(NUM_CLASS)
  # Some `Estimators` use feature columns for understanding their input
  # features.
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


# ======================================
# Auxilliary


def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""
  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]


def super_obtain_results(input_fn, BATCH_SIZE, TRAIN_STEPS,
                         LOG_DIR, LOG_TLE, logger=None):
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

    '''
    if logger:
      logger.critical("\n\n")
      logger.critical("-----------")
      logger.critical("")
      logger.critical("Loss:     {}".format(results["average_loss"]))
      logger.critical("Accuracy: {}".format(results["accuracy"]))
      time_elapsed = time.time() - since
      logger.critical("estimator.config.tf_random_seed = {}".format(
                      estimator.config.tf_random_seed))

      logger.critical("{:17s}".format(experiment_name))
      logger.critical("{:17s} starts at {:s}".format(
          '', time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(since))))
      logger.critical("{:17s} finish at {:s}".format(
          '', time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(time.time()))))
      logger.critical("{:17s} completed at {:.0f}m {:.2f}s".format(
          '', time_elapsed // 60, time_elapsed % 60))
      logger.critical(
          "The entire duration is: {:.6f} min".format(time_elapsed / 60))
      logger.critical("Saved location:")
      logger.critical("\tLOG_DIR: {:s}".format(LOG_DIR))
      logger.critical("\tLOG_TLE: {:s}".format(LOG_TLE))
      logger.critical("")
      logger.critical("-----------\n")

    else:
      print("\n\n-----------\n")
      print("Loss:     {}".format(results["average_loss"]))
      print("Accuracy: {}".format(results["accuracy"]))
      time_elapsed = time.time() - since
      print("estimator.config.tf_random_seed = {}".format(
          estimator.config.tf_random_seed))
      print("{:17s}".format(experiment_name))
      print("{:17s} starts at {:s}".format(
          '', time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(since))))
      print("{:17s} completed at {:.0f}m {:.2f}s".format(
          '', time_elapsed // 60, time_elapsed % 60))
    '''

    time_elapsed = time.time() - since
    csv_temp = time_elapsed / 60  # minutes

    if logger:
      logger.critical("\n\n\n")
      logger.critical("Loss:     {}".format(results["average_loss"]))
      logger.critical("Accuracy: {}".format(results["accuracy"]))
      logger.critical("{:17s} completed in {:.0f} min {:.2f}s".format(
          '', time_elapsed // 60, time_elapsed % 60))
    else:
      print("\n\n\n")
      print("Loss:     {}".format(results["average_loss"]))
      print("Accuracy: {}".format(results["accuracy"]))
      print("{:17s} completed in {:.0f} min {:.2f}s".format(
          '', time_elapsed // 60, time_elapsed % 60))

    return results['average_loss'], results['accuracy'], csv_temp
  return obtain_results
