# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Loss unscaling must avoid overflowing a representable reciprocal."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from jmp._src import loss_scale


class UnscaleReciprocalTest(parameterized.TestCase):

  @parameterized.product(
      cls=(loss_scale.StaticLossScale, loss_scale.DynamicLossScale),
      compiled=(False, True),
      scale=(1e-5, 2e-5),
  )
  def test_small_half_scale_unscales_finite_values(self, cls, compiled, scale):
    small = jnp.asarray(scale, dtype=jnp.float16)
    value = jnp.asarray(.1, dtype=jnp.float16)
    state = cls(small)
    fn = jax.jit(lambda s, x: s.unscale(x)) if compiled else lambda s, x: s.unscale(x)
    result = fn(state, value)
    expected = value / small
    self.assertEqual(result.dtype, jnp.float16)
    self.assertTrue(bool(jnp.isfinite(result)))
    np.testing.assert_allclose(result, expected, rtol=.01)

  @parameterized.product(
      cls=(loss_scale.StaticLossScale, loss_scale.DynamicLossScale),
      dtype=(jnp.float16, jnp.float32, jnp.bfloat16),
  )
  def test_nested_tree_retains_dtype_and_values(self, cls, dtype):
    state = cls(jnp.asarray(2.0, dtype=dtype))
    values = {'weight': jnp.asarray([3., 5.], dtype=dtype),
              'bias': jnp.asarray(.25, dtype=dtype)}
    result = jax.jit(lambda x: state.unscale(x))(values)
    for key in values:
      self.assertEqual(result[key].dtype, values[key].dtype)
      np.testing.assert_allclose(result[key], values[key] / state.loss_scale,
                                 rtol=.01)

  @parameterized.parameters(loss_scale.StaticLossScale, loss_scale.DynamicLossScale)
  def test_float32_gradient_with_half_scale_is_finite(self, cls):
    state = cls(jnp.asarray(1e-5, dtype=jnp.float16))
    value = jnp.asarray(.1, dtype=jnp.float32)
    fn = jax.jit(jax.value_and_grad(lambda x: state.unscale(x)))
    result, grad = fn(value)
    np.testing.assert_allclose(result, value / state.loss_scale, rtol=.001)
    np.testing.assert_allclose(grad, 1. / state.loss_scale.astype(jnp.float32),
                               rtol=.001)
    self.assertTrue(bool(jnp.isfinite(grad)))

  def test_wide_dtype_reference_values_remain_unchanged(self):
    x = jnp.asarray([.01, .1, 1.0], dtype=jnp.float32)
    scale = jnp.asarray(.25, dtype=jnp.float32)
    np.testing.assert_allclose(
        loss_scale.StaticLossScale(scale).unscale(x), x / scale
    )


if __name__ == '__main__':
  absltest.main()
