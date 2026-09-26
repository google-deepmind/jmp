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
"""Tests for jmp._src.loss_scale."""

import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jmp._src import loss_scale as jmp
import numpy as np


class LossScaleTest(parameterized.TestCase):

  def test_no_op_loss_scale(self):
    loss_scale = jmp.NoOpLossScale()
    tree = {"a": jnp.ones([])}
    self.assertIs(loss_scale.scale(tree), tree)
    self.assertIs(loss_scale.unscale(tree), tree)

  @parameterized.named_parameters(
      ("StaticLossScale(2)", jmp.StaticLossScale, 2),
      ("StaticLossScale(3)", jmp.StaticLossScale, 3),
      ("StaticLossScale(4)", jmp.StaticLossScale, 4),
      ("DynamicLossScale(2)", jmp.DynamicLossScale, 2.),
      ("DynamicLossScale(3)", jmp.DynamicLossScale, 3.),
      ("DynamicLossScale(4)", jmp.DynamicLossScale, 4.),
  )
  def test_static_loss_scale(self, cls, scale):
    loss_scale = cls(scale)
    tree = {"a": jnp.array(1.)}
    scaled_tree = {"a": jnp.array(1. * scale)}
    self.assertEqual(loss_scale.scale(tree), scaled_tree)
    self.assertEqual(loss_scale.unscale(scaled_tree), tree)

  @parameterized.named_parameters(
      ("NoOpLossScale", jmp.NoOpLossScale),
      ("StaticLossScale", lambda: jmp.StaticLossScale(0)),  # pytype: disable=wrong-arg-types  # jax-ndarray
  )
  def test_static_empty_trees(self, create):
    loss_scale = create()
    self.assertEmpty(jax.tree_util.tree_leaves(loss_scale))

  def test_dynamic_loss_scale_no_warnings(self):
    with warnings.catch_warnings(record=True) as logged_warnings:
      jmp.DynamicLossScale(2. ** 15)  # pytype: disable=wrong-arg-types  # jax-ndarray
    self.assertEmpty(logged_warnings)

  def test_dynamic_loss_scale_tree(self):
    scale = jnp.ones([])
    counter = jnp.zeros([], jnp.int32)
    period = 2000
    factor = 2
    loss_scale = jmp.DynamicLossScale(scale, counter, period, factor)
    self.assertEqual(jax.tree_util.tree_leaves(loss_scale),
                     [scale, counter, loss_scale.min_loss_scale])
    self.assertEqual(jax.tree_util.tree_map(lambda x: x, loss_scale),
                     loss_scale)

  @parameterized.parameters((20, 2), (30, 3))
  def test_dynamic_loss_scale_adjust_increases_on_finite(self, period, factor):
    grads_finite = jnp.bool_(True)
    loss_scale = jmp.DynamicLossScale(jnp.float32(10), jnp.int32(0),
                                      period, factor)
    for i in range(1, period):
      loss_scale = loss_scale.adjust(grads_finite)
      self.assertEqual(loss_scale.loss_scale, 10)
      self.assertEqual(loss_scale.counter, i)
      self.assertEqual(loss_scale.period, period)
      self.assertEqual(loss_scale.factor, factor)

    # Loss scale should wrap.
    loss_scale = loss_scale.adjust(grads_finite)
    self.assertEqual(loss_scale.loss_scale, 10 * factor)
    self.assertEqual(loss_scale.counter, 0)
    self.assertEqual(loss_scale.period, period)
    self.assertEqual(loss_scale.factor, factor)

  @parameterized.parameters((20, 2), (30, 3))
  def test_dynamic_loss_scale_adjust_reduce_on_non_finite(self, period, factor):
    grads_finite = jnp.bool_(False)
    init = np.float32(10)
    loss_scale = jmp.DynamicLossScale(jnp.asarray(init), jnp.int32(0), period,
                                      factor)
    self.assertLess(init / (factor ** 100), 1, msg="should cover max(1, S)")
    for i in range(100):
      loss_scale = loss_scale.adjust(grads_finite)
      np.testing.assert_allclose(loss_scale.loss_scale,
                                 max(1, init / (factor ** (i + 1))),
                                 rtol=1e-5)
      self.assertEqual(loss_scale.counter, 0)
      self.assertEqual(loss_scale.period, period)
      self.assertEqual(loss_scale.factor, factor)

  @parameterized.parameters((20, 2, .3125), (30, 3, .37), (5., 2., 0.))
  def test_dynamic_loss_scale_explicit_min_loss_scale(self, period, factor,
                                                      min_loss_scale):
    grads_finite = jnp.bool_(False)
    init = np.float32(10)
    loss_scale = jmp.DynamicLossScale(
        jnp.asarray(init), jnp.int32(0), period, factor,
        jnp.asarray(min_loss_scale))
    self.assertLess(init / (factor**100), 1, msg="should cover max(1, S)")
    for i in range(100):
      loss_scale = loss_scale.adjust(grads_finite)
      np.testing.assert_allclose(
          loss_scale.loss_scale,
          max(min_loss_scale, init / (factor**(i + 1))),
          rtol=1e-5)
      self.assertEqual(loss_scale.counter, 0)
      self.assertEqual(loss_scale.period, period)
      self.assertEqual(loss_scale.factor, factor)

  def test_dynamic_loss_scale_adjust_requires_scalar_input(self):
    pass

  def test_dynamic_loss_scale_raises_type_error_on_int_loss_scale(self):
    expected_message = "Expected floating type for loss_scale"
    with self.assertWarnsRegex(Warning, expected_message):
      jmp.DynamicLossScale(jnp.asarray(1, dtype=jnp.int32))

  def test_dynamic_loss_scale_raises_type_error_on_int_min_loss_scale(self):
    expected_message = "Expected floating type for min_loss_scale"
    with self.assertWarnsRegex(Warning, expected_message):
      jmp.DynamicLossScale(jnp.asarray(1, dtype=jnp.float32),
                           min_loss_scale=jnp.asarray(1, dtype=jnp.int32))

  @parameterized.parameters(jnp.inf, jnp.nan)
  def test_all_finite(self, non_finite):
    self.assertTrue(jmp.all_finite(None))
    self.assertTrue(jmp.all_finite({}))
    self.assertFalse(jmp.all_finite({"a": jnp.array(non_finite)}))
    self.assertFalse(jmp.all_finite({"a": jnp.ones([]),
                                     "b": jnp.array(non_finite)}))
    self.assertFalse(jmp.all_finite({"a": jnp.array(non_finite),
                                     "b": jnp.ones([])}))
    self.assertTrue(jmp.all_finite({"a": jnp.ones([]), "b": jnp.ones([])}))

  def test_select_tree(self):
    a = {"a": jnp.ones([]), "b": jnp.zeros([])}
    b = {"a": jnp.zeros([]), "b": jnp.ones([])}
    self.assertIsNone(jmp.select_tree(jnp.bool_(True), None, None))
    self.assertIsNone(jmp.select_tree(jnp.bool_(False), None, None))
    self.assertEqual(jmp.select_tree(jnp.bool_(True), a, b), a)
    self.assertEqual(jmp.select_tree(jnp.bool_(False), a, b), b)

  def test_select_tree_rejects_non_scalar(self):
    with self.assertRaisesRegex(AssertionError, "expected boolean scalar"):
      jmp.select_tree(jnp.ones([1]), None, None)


class MinimumLossScaleTreeTest(parameterized.TestCase):

  def make_scale(self, minimum):
    return jmp.DynamicLossScale(
        loss_scale=jnp.array(4.0, jnp.float32),
        counter=jnp.array(1, jnp.int32),
        period=3,
        factor=2,
        min_loss_scale=jnp.array(minimum, jnp.float32),
    )

  @parameterized.parameters(0.0, 0.125, 4.0, 8.0)
  def test_round_trip_preserves_custom_minimum(self, minimum):
    scale = self.make_scale(minimum)
    leaves, treedef = jax.tree_util.tree_flatten(scale)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(restored.min_loss_scale, minimum)
    self.assertEqual(restored.min_loss_scale.dtype, scale.min_loss_scale.dtype)
    self.assertEqual(restored.loss_scale, scale.loss_scale)
    self.assertEqual(restored.counter, scale.counter)
    self.assertEqual((restored.period, restored.factor), (3, 2))
    self.assertEqual(jax.tree_util.tree_map(lambda x: x, scale), scale)

  @parameterized.parameters(0.0, 0.125, 4.0, 8.0)
  def test_jit_adjust_matches_eager_for_custom_minimum(self, minimum):
    scale = self.make_scale(minimum)
    adjust = jax.jit(lambda s, finite: s.adjust(finite))
    eager, compiled = scale, scale
    for finite in (False, False, False, True, True, True, False):
      eager = eager.adjust(jnp.array(finite))
      compiled = adjust(compiled, jnp.array(finite))
      self.assertEqual(compiled, eager)
      self.assertEqual(compiled.min_loss_scale, minimum)
    self.assertEqual(scale.min_loss_scale, minimum)
    self.assertEqual(scale.loss_scale, 4.0)

  def test_jit_return_preserves_minimum_created_inside_function(self):
    @jax.jit
    def make_and_adjust(minimum):
      scale = self.make_scale(minimum)
      return scale.adjust(jnp.array(False))

    for minimum in (0.125, 4.0, 8.0):
      actual = make_and_adjust(jnp.array(minimum))
      self.assertEqual(actual.min_loss_scale, minimum)
      self.assertEqual(actual.loss_scale, max(2.0, minimum))

  def test_scan_matches_eager_updates_and_preserves_minimum(self):
    finite = jnp.array([False, False, False, True, True, True, False])
    scale = self.make_scale(4.0)

    def step(state, grads_finite):
      new_state = state.adjust(grads_finite)
      return new_state, new_state.loss_scale

    expected = []
    state = scale
    for flag in finite:
      state = state.adjust(flag)
      expected.append(state.loss_scale)
    for run in (
        lambda s: jax.lax.scan(step, s, finite),
        jax.jit(lambda s: jax.lax.scan(step, s, finite)),
    ):
      final, values = run(scale)
      np.testing.assert_array_equal(values, expected)
      self.assertEqual(final, state)
      self.assertEqual(final.min_loss_scale, 4.0)

  def test_vmap_keeps_a_distinct_minimum_for_each_scale(self):
    minima = jnp.array([0.125, 1.0, 4.0, 8.0])
    scales = jax.vmap(self.make_scale)(minima)
    np.testing.assert_array_equal(scales.min_loss_scale, minima)
    adjust = jax.jit(jax.vmap(lambda s: s.adjust(jnp.array(False))))
    adjusted = adjust(scales)
    np.testing.assert_array_equal(adjusted.min_loss_scale, minima)
    np.testing.assert_array_equal(adjusted.loss_scale, [2.0, 2.0, 4.0, 8.0])
    np.testing.assert_array_equal(adjusted.counter, np.zeros(4))

  def test_select_tree_selects_the_minimum_along_with_other_state(self):
    first, second = self.make_scale(0.125), self.make_scale(8.0)
    select = jax.jit(jmp.select_tree)
    for predicate, expected in ((True, first), (False, second)):
      selected = select(jnp.array(predicate), first, second)
      self.assertEqual(selected, expected)

  def test_minimum_is_a_dynamic_argument_to_a_compiled_update(self):
    trace_minima = []

    @jax.jit
    def adjust(scale):
      trace_minima.append(scale.min_loss_scale)
      return scale.adjust(jnp.array(False))

    for minimum in (4.0, 8.0):
      actual = adjust(self.make_scale(minimum))
      self.assertEqual(actual.loss_scale, minimum)
      self.assertEqual(actual.min_loss_scale, minimum)
    self.assertLen(trace_minima, 1)


if __name__ == "__main__":
  absltest.main()
