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
"""Utilities for loss scaling."""

import dataclasses
import functools
from typing import Tuple, TypeVar, Union
import warnings

import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np

# from deepmind.internal import usage_logging

T = TypeVar("T")


def register_empty_pytree(cls):
  tree_util.register_pytree_node(cls, lambda x: ((), x), lambda x, _: x)


@dataclasses.dataclass(frozen=True)
class NoOpLossScale:
  """No-op loss scale does nothing."""

  @property
  def loss_scale(self):
    return 1

  def scale(self, tree: T) -> T:
    # usage_logging.log_event(usage_logging.Event.JMP, "NoOpLossScale")
    return tree

  def unscale(self, tree: T) -> T:
    return tree

  def adjust(self, grads_finite: jnp.ndarray):
    del grads_finite
    return self


@dataclasses.dataclass(frozen=True)
class StaticLossScale:
  """Scales and unscales by a fixed constant."""

  loss_scale: jnp.ndarray

  def scale(self, tree: T) -> T:
    # usage_logging.log_event(usage_logging.Event.JMP, "StaticLossScale")
    return jax.tree_util.tree_map(lambda x: x * self.loss_scale, tree)

  def unscale(self, tree: T) -> T:
    inv_loss_scale = 1 / self.loss_scale
    return jax.tree_util.tree_map(lambda x: x * inv_loss_scale, tree)

  def adjust(self, grads_finite: jnp.ndarray):
    del grads_finite
    return self

_Data = Tuple[jnp.ndarray, ...]
_Meta = Tuple[int, int]


@dataclasses.dataclass(frozen=True)
class DynamicLossScale:
  """Dynamic loss scale.

  Dynamic loss scaling tries to determine the largest loss scale value that
  will keep gradients finite. It does this by increasing the loss scale every
  `period` steps by `factor` if the grads remain finite, otherwise it reduces
  the loss scale by `1 / factor` and resets the counter.

      loss_scale = 2 ** 15
      counter = 0
      period = 2000
      factor = 2

      for step in range(num_steps):
        loss *= loss_scale
        grads /= loss_scale
        grads_finite = all_finite(grads)

        if grads_finite:
          counter += 1
          if counter == period:
            counter = 0
            loss_scale = first_finite(loss_scale * factor, loss_scale)
        else:
          counter = 0
          loss_scale = max(1, loss_scale / factor)

  Typical usage of this class will be something like:

  >>> loss_scale = jmp.DynamicLossScale(jnp.asarray(2 ** 15))
  >>> for _ in range(num_steps):
  ...   # compute loss
  ...   loss = loss_scale.scale(loss)
  ...   # compute grads
  ...   grads = loss_scale.unscale(grads)
  ...   grads_finite = jmp.all_finite(grads)
  ...   loss_scale = loss_scale.adjust(grads_finite)
  ...   # conditionally update params using grads
  """
  loss_scale: jnp.ndarray
  counter: jnp.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros([], np.int32))
  period: int = 2000
  factor: int = 2
  min_loss_scale: jnp.ndarray = dataclasses.field(
      default_factory=lambda: np.ones([], np.int32))

  def __post_init__(self) -> None:
    warn_if_not_floating(self.loss_scale, "loss_scale")
    warn_if_not_floating(self.min_loss_scale, "min_loss_scale")

  def scale(self, tree: T) -> T:
    # usage_logging.log_event(usage_logging.Event.JMP, "DynamicLossScale")
    return jax.tree_util.tree_map(lambda x: x * self.loss_scale, tree)

  def unscale(self, tree: T) -> T:
    inv_loss_scale = 1 / self.loss_scale
    return jax.tree_util.tree_map(lambda x: x * inv_loss_scale, tree)

  def tree_flatten(self) -> Tuple[_Data, _Meta]:
    data = (self.loss_scale, self.counter)
    meta = (self.period, self.factor)
    return data, meta

  @classmethod
  def tree_unflatten(cls, meta: _Meta, data: _Data) -> "DynamicLossScale":
    loss_scale, counter = data
    period, factor = meta
    return cls(loss_scale, counter, period, factor)

  def adjust(self, grads_finite: jnp.ndarray) -> "DynamicLossScale":
    """Returns the next state dependent on whether grads are finite."""
    assert grads_finite.ndim == 0, "Expected boolean scalar"

    first_finite = lambda a, b: jax.lax.select(jnp.isfinite(a).all(), a, b)
    loss_scale = jax.lax.select(
        grads_finite,

        # When grads are finite increase loss scale periodically.
        jax.lax.select(
            self.counter == (self.period - 1),
            first_finite(self.loss_scale * self.factor,
                         self.loss_scale),
            self.loss_scale),

        # If grads are non finite reduce loss scale.
        jnp.maximum(self.min_loss_scale, self.loss_scale / self.factor))

    counter = ((self.counter + 1) % self.period) * grads_finite

    return DynamicLossScale(
        loss_scale=loss_scale,
        counter=counter,
        period=self.period,
        factor=self.factor,
        min_loss_scale=self.min_loss_scale)


register_empty_pytree(NoOpLossScale)
register_empty_pytree(StaticLossScale)
tree_util.register_pytree_node_class(DynamicLossScale)

LossScale = Union[NoOpLossScale, StaticLossScale, DynamicLossScale]


def all_finite(tree) -> jnp.ndarray:
  """Returns a scalar ndarray indicating whether the input arrays are finite."""
  leaves = jax.tree_leaves(tree)
  if not leaves:
    return jnp.array(True)
  else:
    leaves = map(jnp.isfinite, leaves)
    leaves = map(jnp.all, leaves)
    return jnp.stack(list(leaves)).all()


def select_tree(pred: jnp.ndarray, a: T, b: T) -> T:
  """Selects a pytree based on the given predicate."""
  assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
  return jax.tree_map(functools.partial(jax.lax.select, pred), a, b)


def warn_if_not_floating(x: Union[jnp.ndarray, object], var_name: str) -> None:
  """Produces a warning if the given array does not have a floating type.

  This function handles an edgecase where Jax passes in an `object()` to
  determine the structure of user defined pytrees during compilation. They
  recommend explicitly checking if the array in question has the type `object`.

  From the Jax documentation: "The __init__ and __new__ methods of custom
  PyTree classes should generally avoid doing any array conversion or other
  input validation, or else anticipate and handle these special cases."

  See:
  https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

  Args:
    x: Any object.
    var_name: A useful name to put in error messages.
  """
  if type(x) is object:  # pylint: disable=unidiomatic-typecheck
    return
  x_dtype = jax.eval_shape(lambda: x).dtype
  if not jnp.issubdtype(x_dtype, jnp.floating):
    warnings.warn(f"Expected floating type for {var_name}, got {x_dtype}")
