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
"""Utilities for mixed precision in JAX."""

import dataclasses
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
  def conditional_cast(x):
    if (isinstance(x, (np.ndarray, jnp.ndarray)) and
        jnp.issubdtype(x.dtype, jnp.floating)):
      x = x.astype(dtype)
    return x
  return jax.tree_util.tree_map(conditional_cast, tree)


@dataclasses.dataclass(frozen=True)
class Policy:
  """Encapsulates casting for inputs, outputs and parameters."""
  param_dtype: jnp.dtype
  compute_dtype: jnp.dtype
  output_dtype: jnp.dtype

  def cast_to_param(self, x):
    """Converts floating point values to the param dtype."""
    return _cast_floating_to(x, self.param_dtype)

  def cast_to_compute(self, x):
    """Converts floating point values to the compute dtype."""
    return _cast_floating_to(x, self.compute_dtype)

  def cast_to_output(self, x):
    """Converts floating point values to the output dtype."""
    return _cast_floating_to(x, self.output_dtype)

  def with_output_dtype(self, output_dtype: jnp.dtype) -> "Policy":
    return dataclasses.replace(self, output_dtype=output_dtype)

  def __str__(self):
    return "p={},c={},o={}".format(dtype_to_names[self.param_dtype][0],
                                   dtype_to_names[self.compute_dtype][0],
                                   dtype_to_names[self.output_dtype][0])


def get_policy(policy_name: str) -> Policy:
  """Returns a mixed precision policy parsed from a string."""
  # Loose grammar supporting:
  #  - "c=f16"         (params full, compute+output in f16),
  #  - "p=f16,c=f16"   (params, compute and output in f16).
  #  - "p=f16,c=bf16"  (params in f16, compute in bf16, output in bf16)
  # For values that are not specified params defaults to f32, compute follows
  # params and output follows compute (e.g. 'c=f16' -> 'p=f32,c=f16,o=f16').
  param_dtype = jnp.float32
  compute_dtype = output_dtype = None
  if "=" in policy_name:
    for part in policy_name.split(","):
      key, value = part.split("=", 2)
      value = parse_dtype(value)
      if key == "p" or key == "params":
        param_dtype = value
      elif key == "c" or key == "compute":
        compute_dtype = value
      elif key == "o" or key == "output":
        output_dtype = value
      else:
        raise ValueError(f"Unknown key '{key}' in '{policy_name}' should be "
                         "'params', 'compute' or 'output'.")
    if compute_dtype is None:
      compute_dtype = param_dtype
    if output_dtype is None:
      output_dtype = compute_dtype
  else:
    # Assume policy name is a dtype (e.g. 'f32' or 'half') that all components
    # of the policy should contain.
    param_dtype = compute_dtype = output_dtype = parse_dtype(policy_name)

  return Policy(param_dtype=param_dtype, compute_dtype=compute_dtype,
                output_dtype=output_dtype)


def cast_to_full(tree: T) -> T:
  """Ensures floating point leaves of the given tree are f32."""
  return _cast_floating_to(tree, jnp.float32)


def cast_to_half(tree: T) -> T:
  """Ensures floating point leaves of the given tree are half precision."""
  return _cast_floating_to(tree, half_dtype())


def half_dtype() -> jnp.dtype:
  """Returns the half precision dtype for the current backend."""
  device0 = jax.local_devices()[0]
  on_tpu = device0.platform == "tpu"
  return jnp.bfloat16 if on_tpu else jnp.float16


dtype_to_names = {
    jnp.bfloat16: ("bf16", "bfloat16"),
    jnp.float16: ("f16", "float16"),
    jnp.float32: ("full", "f32", "float32"),
    jnp.float64: ("f64", "float64"),
}

name_to_dtype = {name: dtype for dtype, names in dtype_to_names.items()  # pylint: disable=g-complex-comprehension
                 for name in names}


def parse_dtype(value: str) -> jnp.dtype:
  """Parses a string representing a dtype into a dtype object."""
  if value == "half":
    return half_dtype()

  try:
    return name_to_dtype[value]
  except KeyError as e:
    raise ValueError(
        f"Unknown dtype '{value}' must be full,half,float16,bfloat16 or a "
        "contraction thereof (e.g. 'f' for 'full', 'bf16' for 'bfloat16')"
    ) from e
