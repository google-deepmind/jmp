# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""JMP is a Mixed Precision library for JAX."""

from jmp._src.loss_scale import all_finite
from jmp._src.loss_scale import DynamicLossScale
from jmp._src.loss_scale import LossScale
from jmp._src.loss_scale import NoOpLossScale
from jmp._src.loss_scale import select_tree
from jmp._src.loss_scale import StaticLossScale
from jmp._src.policy import cast_to_full
from jmp._src.policy import cast_to_half
from jmp._src.policy import get_policy
from jmp._src.policy import half_dtype
from jmp._src.policy import Policy

__version__ = "0.0.3"

__all__ = (
    "all_finite",
    "DynamicLossScale",
    "LossScale",
    "NoOpLossScale",
    "select_tree",
    "StaticLossScale",
    "cast_to_full",
    "cast_to_half",
    "get_policy",
    "half_dtype",
    "Policy",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the JMP public API.     /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
