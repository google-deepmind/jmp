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
"""Tests for jmp._src.policy."""

import itertools as it
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jmp._src import policy as jmp
import numpy as np

HALF_DTYPES = (np.float16, jnp.float16, jnp.bfloat16)
FULL_DTYPES = (np.float32, jnp.float32)
DTYPES = HALF_DTYPES + FULL_DTYPES
NUMPYS = (np, jnp)


def get_dtype_name(dtype):
  names = {
      np.float16: "float16",
      jnp.bfloat16: "bfloat16",
      np.float32: "float32"
  }
  return names[dtype]


def current_platform():
  return jax.local_devices()[0].platform


def skip_if_unsupported(dtype):
  platform = current_platform()
  if ((platform == "gpu" and dtype == jnp.bfloat16) or
      (platform == "tpu" and dtype in (np.float16, jnp.float16))):
    raise unittest.SkipTest(
        f"{get_dtype_name(dtype)} not supported on {platform}")


class PolicyTest(parameterized.TestCase):

  def assert_dtypes_equal(self, tree_a, tree_b):
    jax.tree_map(lambda a, b: self.assertEqual(a.dtype, b.dtype), tree_a,
                 tree_b)

  @parameterized.parameters(*it.product(DTYPES, NUMPYS))
  def test_policy_cast_to_param(self, dtype, np_):
    skip_if_unsupported(dtype)
    policy = jmp.Policy(dtype, dtype, dtype)
    self.assertEqual(policy.param_dtype, dtype)
    tree = {"a": np_.ones([])}
    self.assert_dtypes_equal(policy.cast_to_param(tree),
                             {"a": np_.ones([], dtype)})

  @parameterized.parameters(*it.product(DTYPES, NUMPYS))
  def test_policy_cast_to_compute(self, dtype, np_):
    skip_if_unsupported(dtype)
    policy = jmp.Policy(dtype, dtype, dtype)
    self.assertEqual(policy.compute_dtype, dtype)
    tree = {"a": np_.ones([])}
    self.assert_dtypes_equal(policy.cast_to_compute(tree),
                             {"a": np_.ones([], dtype)})

  @parameterized.parameters(*it.product(DTYPES, NUMPYS))
  def test_policy_cast_to_output(self, dtype, np_):
    skip_if_unsupported(dtype)
    policy = jmp.Policy(dtype, dtype, dtype)
    self.assertEqual(policy.output_dtype, dtype)
    tree = {"a": np_.ones([])}
    self.assert_dtypes_equal(policy.cast_to_output(tree),
                             {"a": np_.ones([], dtype)})

  @parameterized.parameters(*it.product(DTYPES, NUMPYS))
  def test_policy_with_output_dtype(self, dtype, np_):
    policy = jmp.Policy(np_.float32, np_.float32, np_.float32)
    policy = policy.with_output_dtype(dtype)
    self.assertEqual(policy.output_dtype, dtype)

  @parameterized.parameters(("float16", np.float16),
                            ("float32", np.float32),
                            ("bfloat16", jnp.bfloat16))
  def test_get_policy(self, dtype_name, dtype):
    policy = jmp.get_policy(dtype_name)
    self.assertEqual(policy.param_dtype, dtype)
    self.assertEqual(policy.compute_dtype, dtype)
    self.assertEqual(policy.output_dtype, dtype)

  def test_get_policy_almost_dtype(self):
    with self.assertRaisesRegex(ValueError, "Unknown dtype"):
      jmp.get_policy("compute_float16")

  @parameterized.parameters(*it.product(DTYPES, NUMPYS))
  def test_get_policy_mixed(self, dtype, np_):
    full = np_.float32
    policy = jmp.get_policy(f"c={get_dtype_name(dtype)}")
    self.assertEqual(policy.param_dtype, full)
    self.assertEqual(policy.compute_dtype, dtype)
    self.assertEqual(policy.output_dtype, dtype)

  @parameterized.parameters(*it.product(DTYPES, NUMPYS))
  def test_get_policy_compute(self, dtype, np_):
    full = np_.float32
    policy = jmp.get_policy(f"c={get_dtype_name(dtype)},o=full")
    self.assertEqual(policy.param_dtype, full)
    self.assertEqual(policy.compute_dtype, dtype)
    self.assertEqual(policy.output_dtype, full)

  def test_half_dtype(self):
    if current_platform() == "tpu":
      self.assertEqual(jmp.half_dtype(), jnp.bfloat16)
    else:
      self.assertEqual(jmp.half_dtype(), jnp.float16)

  def test_cast_to_full(self):
    half_tree = dict(o=object(),
                     h=jnp.ones([], dtype=jmp.half_dtype()),
                     f=jnp.ones([]),
                     i=jnp.ones([], dtype=jnp.int16))
    full_tree = dict(o=half_tree["o"],
                     h=half_tree["h"].astype(jnp.float32),
                     f=half_tree["f"],
                     i=half_tree["i"])
    self.assertEqual(jmp.cast_to_full(half_tree), full_tree)

  def test_cast_to_half(self):
    dtype = jmp.half_dtype()
    half_tree = dict(o=object(),
                     h=jnp.ones([], dtype=dtype),
                     f=jnp.ones([]),
                     i=jnp.ones([], dtype=jnp.int16))
    full_tree = dict(o=half_tree["o"],
                     h=half_tree["h"],
                     f=half_tree["f"].astype(dtype),
                     i=half_tree["i"])
    self.assertEqual(jmp.cast_to_half(full_tree), half_tree)

  @parameterized.parameters(*it.product(DTYPES))
  def test_str(self, dtype):
    policy = jmp.Policy(dtype, dtype, dtype)
    policy_str = str(policy)
    for str_piece in policy_str.split(","):
      dtype_str = str_piece.split("=")[1]
      self.assertEqual(dtype_str, jmp.dtype_to_names[dtype][0])

if __name__ == "__main__":
  absltest.main()
