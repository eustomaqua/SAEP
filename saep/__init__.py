# Copyright 2020 The AdaNet+USTC Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAEP (based on AdaNet):
Subarchitecture Ensemble Pruning in Neural Architecture Search.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from saep import distributed
from saep import ensemble
from saep import replay
from saep import subnetwork
from saep.autoensemble import AutoEnsembleEstimator
from saep.autoensemble import AutoEnsembleSubestimator
from saep.autoensemble import AutoEnsembleTPUEstimator
from saep.core import Estimator
from saep.core import Evaluator
from saep.core import ReportMaterializer
from saep.core import Summary
from saep.core import TPUEstimator
# For backwards compatibility. Previously all Ensemblers were complexity
# regularized using the AdaNet objective.
from saep.ensemble import ComplexityRegularized as Ensemble
from saep.ensemble import MixtureWeightType
from saep.ensemble import WeightedSubnetwork
from saep.subnetwork import Subnetwork

from saep.version import __version__

__all__ = [
    "AutoEnsembleEstimator",
    "AutoEnsembleSubestimator",
    "AutoEnsembleTPUEstimator",
    "distributed",
    "ensemble",
    "Ensemble",
    "Estimator",
    "Evaluator",
    "replay",
    "ReportMaterializer",
    "subnetwork",
    "Summary",
    "TPUEstimator",
    "MixtureWeightType",
    "WeightedSubnetwork",
    "Subnetwork",
]
