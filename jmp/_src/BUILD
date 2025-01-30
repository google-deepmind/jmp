load("//jmp/_src:build_defs.bzl", "jmp_py_library", "jmp_py_test")

package(
    default_applicable_licenses = ["//jmp:license"],
    default_visibility = ["//jmp:__subpackages__"],
)

licenses(["notice"])

jmp_py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    deps = [
        # pip: jax
        # pip: numpy
    ],
)

jmp_py_test(
    name = "loss_scale_test",
    srcs = ["loss_scale_test.py"],
    deps = [
        ":loss_scale",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)

jmp_py_library(
    name = "policy",
    srcs = ["policy.py"],
    deps = [
        # pip: jax
        # pip: numpy
    ],
)

jmp_py_test(
    name = "policy_test",
    srcs = ["policy_test.py"],
    deps = [
        ":policy",
        # pip: absl/testing:absltest
        # pip: absl/testing:parameterized
        # pip: jax
        # pip: numpy
    ],
)
