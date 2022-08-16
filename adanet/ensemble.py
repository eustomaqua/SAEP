# Copyright 2019 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines built-in ensemble methods and interfaces for custom ensembles."""

# TODO: Add more detailed documentation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from adanet.ensemble_ensembler import Ensemble
from adanet.ensemble_ensembler import Ensembler
from adanet.ensemble_ensembler import TrainOpSpec
from adanet.ensemble_mean import MeanEnsemble
from adanet.ensemble_mean import MeanEnsembler
from adanet.ensemble_strategy import AllStrategy
from adanet.ensemble_strategy import Candidate
from adanet.ensemble_strategy import GrowStrategy
from adanet.ensemble_strategy import SoloStrategy
from adanet.ensemble_strategy import Strategy
from adanet.ensemble_weighted import ComplexityRegularized
from adanet.ensemble_weighted import ComplexityRegularizedEnsembler
from adanet.ensemble_weighted import MixtureWeightType
from adanet.ensemble_weighted import WeightedSubnetwork

__all__ = [
    "Ensemble",
    "Ensembler",
    "TrainOpSpec",
    "AllStrategy",
    "Candidate",
    "GrowStrategy",
    "SoloStrategy",
    "Strategy",
    "ComplexityRegularized",
    "ComplexityRegularizedEnsembler",
    "MeanEnsemble",
    "MeanEnsembler",
    "MixtureWeightType",
    "WeightedSubnetwork",
]
