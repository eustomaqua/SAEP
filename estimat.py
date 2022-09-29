# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow.compat.v1 as tf

import adanet
import adanet.examples_simple_dnn as simple_dnn
from classes import SimpleCNNGenerator


import saep as SAEP
import saep.examples_simple_dnn as pruned_dnn

from classes import PrunedCNNGenerator
# from classes import ComplexCNNGenerator


# ======================================
# Model


class AbstractCreate(object):
  def __init__(self, random_seed):
    self.RANDOM_SEED = random_seed
    # self.varient = variant

  def assign_train_param(self, learning_rate, batch_size, train_steps):
    self.LEARNING_RATE = learning_rate
    self.BATCH_SIZE = batch_size
    self.TRAIN_STEPS = train_steps

  def assign_adanet_para(self, adanet_iterations, adanet_lambda=.99):
    #                      learn_mixture_weights=False):
    self.ADANET_ITERATIONS = adanet_iterations
    self.ADANET_LAMBDA = adanet_lambda
    # self.LEARN_MIXTURE_WEIGHTS = learn_mixture_weights

  def assign_expt_params(self, n_classes, experiment_name, log_dir):
    self.NUM_CLASS = n_classes
    self.experiment_name = experiment_name  # this_experiment
    self.LOG_DIR = log_dir

  def assign_SAEP_adapru(self, ensemble_pruning="keep_all",
                         thinp_alpha=0.5,  # logger=None,
                         final=False):
    self.ensemble_pruning = ensemble_pruning
    self.thinp_alpha = thinp_alpha
    # self.logger = logger
    self.final = final

  def assign_SAEP_logger(self, logger=None):
    self.logger = logger

  def make_config(self, experiment_name, save_steps=1000):
    model_dir = os.path.join(self.LOG_DIR, experiment_name)
    # Estimator configuration.
    return tf.estimator.RunConfig(
        save_checkpoints_steps=save_steps,
        save_summary_steps=save_steps,
        tf_random_seed=self.RANDOM_SEED,
        model_dir=model_dir)

  def train_and_evaluate(self, estimator, input_fn):
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn(
            "train", training=True, batch_size=self.BATCH_SIZE),
        max_steps=self.TRAIN_STEPS)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn(
            "test", training=False, batch_size=self.BATCH_SIZE),
        steps=None,
        start_delay_secs=1,
        throttle_secs=1)
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=train_spec,
        eval_spec=eval_spec)
    results = estimator.evaluate(
        input_fn=input_fn(
            "test", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return results, estimator

  def _new_linear(self, feature_columns, input_fn):
    raise NotImplementedError

  def _new_simple_dnn(self, feature_columns, head, input_fn):
    raise NotImplementedError

  def _new_simple_cnn(self, feature_columns, head, input_fn):
    raise NotImplementedError

  def _new_complex_cnn(self, feature_columns, head, input_fn):
    return NotImplementedError

  def create_estimator(self, modeluse, feature_columns, head, input_fn):
    if modeluse == "linear":
      return self._new_linear(feature_columns, input_fn)
    elif modeluse == "dnn":  # "simple_dnn"
      return self._new_simple_dnn(feature_columns, head, input_fn)
    elif modeluse == "cnn":  # "simple_cnn"
      return self._new_simple_cnn(feature_columns, head, input_fn)
    elif modeluse == "cpx":  # "complex_cnn"
      return self._new_complex_cnn(feature_columns, head, input_fn)
    raise ValueError("invalid `modeluse`, {}.".format(modeluse))

  # for variant
  def ensemble_architecture(self, results):
    """Extracts the ensemble architecture from evaluation."""
    architecture = results["architecture/adanet/ensembles"]
    # The architecture is a serialized Summary proto for TensorBoard.
    summary_proto = tf.summary.Summary.FromString(architecture)
    return summary_proto.value[0].tensor.string_val[0]


# --------------------------------------
# AdaNet


class AdaNetOriginal(AbstractCreate):
  """docstring for AdaNetOriginal"""

  def __init__(self, random_seed):
    super().__init__(random_seed)

  def _new_linear(self, feature_columns, input_fn):
    return tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=self.NUM_CLASS,
        optimizer=tf.train.RMSPropOptimizer(
            learning_rate=self.LEARNING_RATE),
        config=self.make_config(self.experiment_name))

  def _new_simple_dnn(self, feature_columns, head, input_fn):
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=self.LEARNING_RATE)
    subnetwork_generator = simple_dnn.Generator(
        feature_columns=feature_columns,
        optimizer=optimizer,
        seed=self.RANDOM_SEED)
    evaluator = adanet.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return adanet.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=self.TRAIN_STEPS // self.ADANET_ITERATIONS,
        evaluator=evaluator,
        config=self.make_config(self.experiment_name))

  def _new_simple_cnn(self, feature_columns, head, input_fn):
    max_iteration_steps = self.TRAIN_STEPS // self.ADANET_ITERATIONS
    subnetwork_generator = SimpleCNNGenerator(
        learning_rate=self.LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        seed=self.RANDOM_SEED)
    evaluator = adanet.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return adanet.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        evaluator=evaluator,
        adanet_loss_decay=.99,
        config=self.make_config(self.experiment_name))


class AdaNetVariants(AbstractCreate):
  def __init__(self, random_seed):
    super().__init__(random_seed)

  def _new_simple_dnn(self, feature_columns, head, input_fn):
    ensembler_optimizer = None
    if self.LEARN_MIXTURE_WEIGHTS:
      ensembler_optimizer = tf.train.RMSPropOptimizer(
          learning_rate=self.LEARNING_RATE)
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=self.LEARNING_RATE)
    subnetwork_generator = simple_dnn.Generator(
        feature_columns=feature_columns,
        optimizer=optimizer,
        learn_mixture_weights=self.LEARN_MIXTURE_WEIGHTS,
        seed=self.RANDOM_SEED)
    evaluator = adanet.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE))
    return adanet.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=self.TRAIN_STEPS // self.ADANET_ITERATIONS,
        evaluator=evaluator,
        ensemblers=[
            adanet.ensemble.ComplexityRegularizedEnsembler(
                optimizer=ensembler_optimizer,
                adanet_lambda=self.ADANET_LAMBDA),
        ],
        config=self.make_config(self.experiment_name, 5000))

  def _new_simple_cnn(self, feature_columns, head, input_fn):
    ensembler_optimizer = None
    if self.LEARN_MIXTURE_WEIGHTS:
      ensembler_optimizer = tf.train.RMSPropOptimizer(
          learning_rate=self.LEARNING_RATE)
    max_iteration_steps = self.TRAIN_STEPS // self.ADANET_ITERATIONS
    subnetwork_generator = SimpleCNNGenerator(
        learning_rate=self.LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        seed=self.RANDOM_SEED)
    evaluator = adanet.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return adanet.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        evaluator=evaluator,
        ensemblers=[
            adanet.ensemble.ComplexityRegularizedEnsembler(
                optimizer=ensembler_optimizer,
                adanet_lambda=self.ADANET_LAMBDA),
        ],
        adanet_loss_decay=.99,
        config=self.make_config(self.experiment_name, 5000))


# --------------------------------------
# SAEP


class AdaPruOriginal(AbstractCreate):
  """docstring for CreateExpected"""

  def __init__(self, random_seed, type_pruning):
    super().__init__(random_seed)
    self.type_pruning = type_pruning

  def _new_simple_dnn(self, feature_columns, head, input_fn):
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=self.LEARNING_RATE)
    subnetwork_generator = pruned_dnn.Generator(
        feature_columns=feature_columns,
        optimizer=optimizer,
        seed=self.RANDOM_SEED)
    evaluator = SAEP.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return SAEP.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=self.TRAIN_STEPS // self.ADANET_ITERATIONS,
        evaluator=evaluator,
        config=self.make_config(self.experiment_name),
        ensemble_pruning=self.ensemble_pruning,
        adanet_iterations=self.ADANET_ITERATIONS,
        thinp_alpha=self.thinp_alpha,
        logger=self.logger, final=self.final)

  def _new_simple_cnn(self, feature_columns, head, input_fn):
    max_iteration_steps = self.TRAIN_STEPS // self.ADANET_ITERATIONS
    subnetwork_generator = PrunedCNNGenerator(
        learning_rate=self.LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        seed=self.RANDOM_SEED)
    evaluator = SAEP.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return SAEP.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        evaluator=evaluator,
        adanet_loss_decay=.99,
        config=self.make_config(self.experiment_name),
        ensemble_pruning=self.ensemble_pruning,
        adanet_iterations=self.ADANET_ITERATIONS,
        thinp_alpha=self.thinp_alpha,
        logger=self.logger, final=self.final)

  """
  def _new_complex_cnn(self, feature_columns, head, input_fn):
    max_iteration_steps = self.TRAIN_STEPS // self.ADANET_ITERATIONS
    subnetwork_generator = ComplexCNNGenerator(
        learning_rate=self.LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        seed=self.RANDOM_SEED)
    evaluator = SAEP.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return SAEP.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        evaluator=evaluator,
        adanet_loss_decay=.99,
        config=self.make_config(self.experiment_name),
        ensemble_pruning=self.ensemble_pruning,
        adanet_iterations=self.ADANET_ITERATIONS,
        thinp_alpha=self.thinp_alpha,
        logger=self.logger, final=self.final)
  """


class AdaPruVariants(AbstractCreate):
  def __init__(self, random_seed, type_pruning):
    super().__init__(random_seed)
    self.type_pruning = type_pruning

  def _new_simple_dnn(self, feature_columns, head, input_fn):
    ensembler_optimizer = None
    if self.LEARN_MIXTURE_WEIGHTS:
      ensembler_optimizer = tf.train.RMSPropOptimizer(
          learning_rate=self.LEARNING_RATE)
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=self.LEARNING_RATE)
    subnetwork_generator = pruned_dnn.Generator(
        feature_columns=feature_columns,
        optimizer=optimizer,
        learn_mixture_weights=self.LEARN_MIXTURE_WEIGHTS,
        seed=self.RANDOM_SEED)
    evaluator = SAEP.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return SAEP.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=self.TRAIN_STEPS // self.ADANET_ITERATIONS,
        evaluator=evaluator,
        ensemblers=[
            SAEP.ensemble.ComplexityRegularizedEnsembler(
                optimizer=ensembler_optimizer,
                adanet_lambda=self.ADANET_LAMBDA)],
        config=self.make_config(self.experiment_name),
        ensemble_pruning=self.ensemble_pruning,
        adanet_iterations=self.ADANET_ITERATIONS,
        thinp_alpha=self.thinp_alpha,
        logger=self.logger, final=self.final)

  def _new_simple_cnn(self, feature_columns, head, input_fn):
    ensembler_optimizer = None
    if self.LEARN_MIXTURE_WEIGHTS:
      ensembler_optimizer = tf.train.RMSPropOptimizer(
          learning_rate=self.LEARNING_RATE)
    max_iteration_steps = self.TRAIN_STEPS // self.ADANET_ITERATIONS
    subnetwork_generator = PrunedCNNGenerator(
        learning_rate=self.LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        learn_mixture_weights=self.LEARN_MIXTURE_WEIGHTS,
        seed=self.RANDOM_SEED)
    evaluator = SAEP.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return SAEP.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        evaluator=evaluator,
        ensemblers=[
            SAEP.ensemble.ComplexityRegularizedEnsembler(
                optimizer=ensembler_optimizer,
                adanet_lambda=self.ADANET_LAMBDA)],
        adanet_loss_decay=.99,
        config=self.make_config(self.experiment_name),
        ensemble_pruning=self.ensemble_pruning,
        adanet_iterations=self.ADANET_ITERATIONS,
        thinp_alpha=self.thinp_alpha,
        logger=self.logger, final=self.final)

  """
  def _new_complex_cnn(self, feature_columns, head, input_fn):
    ensembler_optimizer = None
    if self.LEARN_MIXTURE_WEIGHTS:
      ensembler_optimizer = tf.train.RMSPropOptimizer(
          learning_rate=self.LEARNING_RATE)
    max_iteration_steps = self.TRAIN_STEPS // self.ADANET_ITERATIONS
    subnetwork_generator = ComplexCNNGenerator(
        learning_rate=self.LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        learn_mixture_weights=self.LEARN_MIXTURE_WEIGHTS,
        seed=self.RANDOM_SEED)
    evaluator = SAEP.Evaluator(
        input_fn=input_fn(
            "train", training=False, batch_size=self.BATCH_SIZE),
        steps=None)
    return SAEP.Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        evaluator=evaluator,
        ensemblers=[
            SAEP.ensemble.ComplexityRegularizedEnsembler(
                optimizer=ensembler_optimizer,
                adanet_lambda=self.ADANET_LAMBDA),
        ],
        adanet_loss_decay=.99,
        config=self.make_config(self.experiment_name),
        ensemble_pruning=self.ensemble_pruning,
        adanet_iterations=self.ADANET_ITERATIONS,
        thinp_alpha=self.thinp_alpha,
        logger=self.logger, final=self.final)
  """
