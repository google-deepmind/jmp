# Mixed precision training in [JAX]

![Test status](https://github.com/deepmind/jmp/workflows/pytest/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/jmp)

[**Installation**](#installation)
| [**Examples**](#examples)
| [**Policies**](#policies)
| [**Loss scaling**](#loss-scaling)
| [**Citing JMP**](#citing-jmp)
| [**References**](#references)

Mixed precision training [[0]] is a technique that mixes the use of full and
half precision floating point numbers during training to reduce the memory
bandwidth requirements and improve the computational efficiency of a given
model.

This library implements support for mixed precision training in [JAX] by providing
two key abstractions (mixed precision "policies" and loss scaling). Neural
network libraries (such as [Haiku]) can integrate with `jmp` and provide
"Automatic Mixed Precision (AMP)" support (automating or simplifying applying
policies to modules).

All code examples below assume the following:

```python
import jax
import jax.numpy as jnp
import jmp

half = jnp.float16  # On TPU this should be jnp.bfloat16.
full = jnp.float32
```

## Installation

JMP is written in pure Python, but depends on C++ code via JAX and NumPy.

Because JAX installation is different depending on your CUDA version, JMP does
not list JAX as a dependency in `requirements.txt`.

First, follow [these instructions](https://github.com/google/jax#installation)
to install JAX with the relevant accelerator support.

Then, install JMP using pip:

```bash
$ pip install git+https://github.com/deepmind/jmp
```

## Examples

You can find a
[fully worked JMP example in Haiku](https://github.com/deepmind/dm-haiku/tree/master/examples/imagenet)
which shows how to use mixed f32/f16 precision to halve training time on GPU and
mixed f32/bf16 to reduce training time on TPU by a third.

## Policies

A mixed precision policy encapsulates the configuration in a mixed precision
experiment.

```python
# Our policy specifies that we will store parameters in full precision but will
# compute and return output in half precision.
my_policy = jmp.Policy(compute_dtype=half,
                       param_dtype=full,
                       output_dtype=half)
```

The policy object can be used to cast pytrees:

```python
def layer(params, x):
  params, x = my_policy.cast_to_compute((params, x))
  w, b = params
  y = x @ w + b
  return my_policy.cast_to_output(y)

params = {"w": jnp.ones([], dtype=my_policy.param_dtype)}
y = layer(params, x)
assert y.dtype == half
```

You can replace the output type of a given policy:

```python
my_policy = my_policy.with_output_dtype(full)
```

You can also define a policy via a string, which may be useful for specifying a
policy as a command-line argument or as a hyperparameter to your experiment:

```python
my_policy = jmp.get_policy("params=float32,compute=float16,output=float32")
float16 = jmp.get_policy("float16")  # Everything in f16.
half = jmp.get_policy("half")        # Everything in half (f16 or bf16).
```

## Loss scaling

When training with reduced precision, consider whether gradients will need to be
shifted into the representable range of the format that you are using. This is
particularly important when training with `float16` and less important for
`bfloat16`. See the NVIDIA mixed precision user guide [[1]] for more details.

The easiest way to shift gradients is with loss scaling, which scales your loss
and gradients by `S` and `1/S` respectively.

```python
def my_loss_fn(params, loss_scale: jmp.LossScale, ...):
  loss = ...
  # You should apply regularization etc before scaling.
  loss = loss_scale.scale(loss)
  return loss

def train_step(params, loss_scale: jmp.LossScale, ...):
  grads = jax.grad(my_loss_fn)(...)
  grads = loss_scale.unscale(grads)
  # You should put gradient clipping etc after unscaling.
  params = apply_optimizer(params, grads)
  return params

loss_scale = jmp.StaticLossScale(2 ** 15)
for _ in range(num_steps):
  params = train_step(params, loss_scale, ...)
```

The appropriate value for `S` depends on your model, loss, batch size and
potentially other factors. You can determine this with trial and error. As a
rule of thumb you want the largest value of `S` that does not introduce overflow
during backprop. NVIDIA [[1]] recommend computing statistics about the gradients
of your model (in full precision) and picking `S` such that its product with the
maximum norm of your gradients is below `65,504`.

We provide a dynamic loss scale, which adjusts the loss scale periodically
during training to find the largest value for `S` that produces finite
gradients. This is more convenient and robust compared with picking a static
loss scale, but has a small performance impact (between 1 and 5%).

```python
def my_loss_fn(params, loss_scale: jmp.LossScale, ...):
  loss = ...
  # You should apply regularization etc before scaling.
  loss = loss_scale.scale(loss)
  return loss

def train_step(params, loss_scale: jmp.LossScale, ...):
  grads = jax.grad(my_loss_fn)(...)
  grads = loss_scale.unscale(grads)
  # You should put gradient clipping etc after unscaling.

  # You definitely want to skip non-finite updates with the dynamic loss scale,
  # but you might also want to consider skipping them when using a static loss
  # scale if you experience NaN's when training.
  skip_nonfinite_updates = isinstance(loss_scale, jmp.DynamicLossScale)

  if skip_nonfinite_updates:
    grads_finite = jmp.all_finite(grads)
    # Adjust our loss scale depending on whether gradients were finite. The
    # loss scale will be periodically increased if gradients remain finite and
    # will be decreased if not.
    loss_scale = loss_scale.adjust(grads_finite)
    # Only apply our optimizer if grads are finite, if any element of any
    # gradient is non-finite the whole update is discarded.
    params = jmp.select_tree(grads_finite, apply_optimizer(params, grads), params)
  else:
    # With static or no loss scaling just apply our optimizer.
    params = apply_optimizer(params, grads)

  # Since our loss scale is dynamic we need to return the new value from
  # each step. All loss scales are `PyTree`s.
  return params, loss_scale

loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15))
for _ in range(num_steps):
  params, loss_scale = train_step(params, loss_scale, ...)
```

In general using a static loss scale should offer the best speed, but we have
optimized dynamic loss scaling to make it competitive. We recommend you start
with dynamic loss scaling and move to static loss scaling if performance is an
issue.

We finally offer a no-op loss scale which you can use as a drop in replacement.
It does nothing (apart from implement the `jmp.LossScale` API):

```python
loss_scale = jmp.NoOpLossScale()
assert loss is loss_scale.scale(loss)
assert grads is loss_scale.unscale(grads)
assert loss_scale is loss_scale.adjust(grads_finite)
assert loss_scale.loss_scale == 1
```

## Citing JMP

This repository is part of the [DeepMind JAX Ecosystem](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research),
to cite JMP please use the [DeepMind JAX Ecosystem citation](https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt).

## References

[[0]] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich
Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh
Venkatesh, Hao Wu: "Mixed Precision Training", 2017; arXiv:1710.03740
https://arxiv.org/abs/1710.03740.

[[1]] "Training With Mixed Precision :: NVIDIA Deep Learning Performance
Documentation". Docs.Nvidia.Com, 2020,
https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/.

[0]: https://arxiv.org/abs/1710.03740
[1]: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
[Haiku]: https://github.com/deepmind/dm-haiku
[JAX]: https://github.com/google/jax
