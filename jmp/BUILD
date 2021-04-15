# Description: JMP is a JAX Mixed Precision library.
package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "jmp",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//jmp/_src:loss_scale",
        "//jmp/_src:policy",
    ],
)
