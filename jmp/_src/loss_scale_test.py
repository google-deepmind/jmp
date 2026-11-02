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
      ("StaticLossScale", lambda: jmp.StaticLossScale(0)),  # pyrefly: ignore[bad-argument-type]
  )
  def test_static_empty_trees(self, create):
    loss_scale = create()
    self.assertEmpty(jax.tree_util.tree_leaves(loss_scale))

  def test_dynamic_loss_scale_no_warnings(self):
    with warnings.catch_warnings(record=True) as logged_warnings:
      jmp.DynamicLossScale(2. ** 15)  # pyrefly: ignore[bad-argument-type]
    self.assertEmpty(logged_warnings)

  def test_dynamic_loss_scale_tree(self):
    scale = jnp.ones([])
    counter = jnp.zeros([], jnp.int32)
    period = 2000
    factor = 2
    loss_scale = jmp.DynamicLossScale(scale, counter, period, factor)
    self.assertEqual(jax.tree_util.tree_leaves(loss_scale), [scale, counter])
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

    # Loss scale should wrap.
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

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32), use_jit=(False, True))
  def test_dynamic_loss_scale_adjust_preserves_dtype(self, dtype, use_jit):
    scale = jmp.DynamicLossScale(dtype(8), period=2)
    adjust = jax.jit(lambda state, flag: state.adjust(flag)) if use_jit else (
        lambda state, flag: state.adjust(flag))
    flags = (True, True, False, False, False, False, False)
    # Two finite steps grow 8 to 16; failures halve it to the default floor 1.
    expected_scales = (8, 16, 8, 4, 2, 1, 1)
    expected_counters = (1, 0, 0, 0, 0, 0, 0)
    for flag, expected, counter in zip(flags, expected_scales, expected_counters):
      scale = adjust(scale, jnp.asarray(flag))
      self.assertEqual(scale.loss_scale.dtype, dtype)
      self.assertEqual(scale.loss_scale, expected)
      self.assertEqual(scale.counter, counter)

  @parameterized.parameters(jnp.float16, jnp.bfloat16, jnp.float32)
  def test_dynamic_loss_scale_low_precision_scan(self, dtype):
    def step(scale, flag):
      scale = scale.adjust(flag)
      return scale, scale.loss_scale

    scale = jmp.DynamicLossScale(dtype(8), period=2)
    flags = jnp.asarray([True, True, False, False, False, False, False])
    final, scales = jax.lax.scan(step, scale, flags)
    self.assertEqual(scales.dtype, dtype)
    np.testing.assert_array_equal(scales, [8, 16, 8, 4, 2, 1, 1])
    self.assertEqual(final.loss_scale, 1)
    self.assertEqual(final.counter, 0)

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32), use_jit=(False, True))
  def test_dynamic_loss_scale_low_precision_overflow(self, dtype, use_jit):
    maximum = jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)
    scale = jmp.DynamicLossScale(maximum, period=1)
    adjust = jax.jit(lambda state: state.adjust(jnp.asarray(True))) if use_jit else (
        lambda state: state.adjust(jnp.asarray(True)))
    updated = adjust(scale)
    self.assertEqual(updated.loss_scale.dtype, dtype)
    self.assertEqual(updated.loss_scale, maximum)
    self.assertEqual(updated.counter, 0)

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

if __name__ == "__main__":
  absltest.main()
