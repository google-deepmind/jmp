load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//tools/build_defs/license:license.bzl", "license")

# Description: JMP is a JAX Mixed Precision library.
package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:private"],
)

license(
    name = "license",
    package_name = "jmp",
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "jmp",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//jmp/_src:loss_scale",
        "//jmp/_src:policy",
    ],
)
